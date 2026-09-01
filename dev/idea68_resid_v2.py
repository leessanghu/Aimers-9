"""idea68 — idea65 재설계: 더 나은 v66프록시 + 정규화 완화.

idea65 실패원인 진단 2가지, 둘 다 고친다:
  1) 잔차타깃이 "y - ability_loo"였다 -> ability_loo는 v66과 corr 0.79 정도뿐인
     약한 프록시. 대신 fold A/C 진짜 v66에 4피처(ability_here/inseason_success/
     cmd_index/reverse_smooth) 선형회귀를 적합해(R^2 0.69/0.63, corr 0.83/0.79)
     train행에 그대로 적용하는 "더 나은 프록시"를 쓴다.
  2) emb_weight_decay가 3e-2로 너무 세서 신호까지 죽었다(corr(진짜잔차)=0.0044)
     -> 3e-3으로 10배 완화. gradient clipping은 유지(공짜 안정화 장치).

judge: 진짜 v66(fold A 캐시)에 대한 residual correction 손익분기.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
CD = "idea68_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
MIN_COUNT = 30
EMB = 8


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
season = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
bid = meta["batter_id"].to_numpy()
Xn = X.to_numpy(np.float32)

UPTO, VAL = 2023, 2024
tr, va = season <= UPTO, season == VAL
yv = y[va]
r = float(yv.mean())
bs = r * (1 - r)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / bs)

avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
base = avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6", "d8", "sub")])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ("d6", "d8")], axis=0)
mr = avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42, 7)])
od = avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42, 7)])
mo = avg([f"idea46_cache/A_midother_s{s}.npy" for s in (42, 7)])
cb = avg([f"idea54_cache/A_cond_ball_s{s}.npy" for s in (42, 7)])
cr = avg([f"idea54_cache/A_count_resid_s{s}.npy" for s in (42, 7)])
f5 = avg([f"idea54_cache/A_future50_multi_s{s}.npy" for s in (42, 7)])
v66_va = (.1824 * base + .2432 * hur + .0608 * mr + .1216 * od + .1520 * mo
          + .08 * cb + .08 * cr + .08 * f5)
B66 = sc(v66_va)
e1 = v66_va - yv
B1 = float(np.mean(e1 ** 2))
log(f"v66local={B66:.2f}  B1={B1:.6f}")

# v66 선형프록시를 valid(fold A)에서 적합해 train 전 구간에 적용
proxy_cols = ["x_ability_here", "inseason_success_smooth", "inseason_cmd_index", "inseason_reverse_smooth"]
Pva = X.loc[va, proxy_cols].to_numpy(np.float64)
Aeq = np.column_stack([np.ones(va.sum()), Pva])
coef, _, _, _ = np.linalg.lstsq(Aeq, v66_va, rcond=None)
r2 = 1 - np.sum((v66_va - Aeq @ coef) ** 2) / np.sum((v66_va - v66_va.mean()) ** 2)
log(f"v66 선형프록시 적합: R^2={r2:.4f} corr={np.corrcoef(v66_va, Aeq@coef)[0,1]:.4f} coef={np.round(coef,4)}")

Pall = X[proxy_cols].to_numpy(np.float64)
v66_proxy_all = np.column_stack([np.ones(len(X)), Pall]) @ coef
resid_target = (y - v66_proxy_all).astype(np.float32)
log(f"잔차타깃(선형v66프록시 기준) train내 mean={resid_target[tr].mean():+.5f} sd={resid_target[tr].std():.4f}")

vc_p = pd.Series(pid[tr]).value_counts()
vc_b = pd.Series(bid[tr]).value_counts()
pmap = {v: i + 1 for i, v in enumerate(vc_p[vc_p >= MIN_COUNT].index)}
bmap = {v: i + 1 for i, v in enumerate(vc_b[vc_b >= MIN_COUNT].index)}
ip = np.array([pmap.get(v, 0) for v in pid], dtype=np.int64)
ib = np.array([bmap.get(v, 0) for v in bid], dtype=np.int64)
mu, sd_ = Xn[tr].mean(0), Xn[tr].std(0)
sd_[sd_ < 1e-8] = 1.0
Z = np.clip((Xn - mu) / sd_, -10, 10)
tr_idx = np.where(tr)[0]
cut = int(len(tr_idx) * 0.92)
fit_i, es_i = tr_idx[:cut], tr_idx[cut:]
va_i = np.where(va)[0]
w_rec = (0.5 ** ((UPTO - season) / 2.0)).astype(np.float32)
T = lambda a, i: torch.as_tensor(a[i])


class Net(nn.Module):
    def __init__(self, n_num, n_p, n_b):
        super().__init__()
        self.ep = nn.Embedding(n_p, EMB)
        self.eb = nn.Embedding(n_b, EMB)
        nn.init.normal_(self.ep.weight, 0, 0.01)
        nn.init.normal_(self.eb.weight, 0, 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(n_num + 2 * EMB, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, xn, xp, xb):
        return self.mlp(torch.cat([xn, self.ep(xp), self.eb(xb)], 1)).squeeze(1)


def train_one(seed, emb_wd, max_ep=40, pat=6, lr=5e-4):
    torch.manual_seed(seed)
    net = Net(Xn.shape[1], len(pmap) + 1, len(bmap) + 1)
    opt = torch.optim.AdamW([
        {"params": list(net.mlp.parameters()), "weight_decay": 1e-5},
        {"params": list(net.ep.parameters()) + list(net.eb.parameters()), "weight_decay": emb_wd},
    ], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    rt = torch.as_tensor(resid_target)
    ez_, ep__, eb__, ey_, ew_ = T(Z, es_i), T(ip, es_i), T(ib, es_i), rt[es_i], T(w_rec, es_i)
    best, best_state, bad = np.inf, None, 0
    for e in range(max_ep):
        net.train()
        permE = np.random.default_rng(seed * 1000 + e).permutation(len(fit_i))
        for s in range(0, len(permE), 8192):
            j = fit_i[permE[s:s + 8192]]
            opt.zero_grad()
            w = T(w_rec, j)
            loss = (((net(T(Z, j), T(ip, j), T(ib, j)) - rt[j]) ** 2) * w).sum() / w.sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float((((net(ez_, ep__, eb__) - ey_) ** 2) * ew_).sum() / ew_.sum())
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    log(f"  [emb_wd={emb_wd} seed={seed}] 학습완료 best_es={best:.6f} epochs={e+1}")
    net.load_state_dict(best_state)
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(va_i), 65536):
            j = va_i[s:s + 65536]
            out.append(net(T(Z, j), T(ip, j), T(ib, j)).numpy())
    return np.concatenate(out).astype(np.float64)


results = []
for emb_wd in (3e-3, 1e-3, 1e-2):
    rhat = train_one(42, emb_wd)
    np.save(f"{CD}/A_embwd{emb_wd}.npy", rhat)
    c_v66 = np.corrcoef(rhat, v66_va)[0, 1]
    c_innov = np.corrcoef(rhat, -e1)[0, 1]
    log(f"  rhat(emb_wd={emb_wd}): sd={rhat.std():.5f} corr(v66예측)={c_v66:.4f} corr(진짜잔차)={c_innov:.4f}")
    fine = np.linspace(-0.2, 0.5, 141)
    vals = [sc(v66_va + a * rhat) - B66 for a in fine]
    j = int(np.argmax(vals))
    log(f"    fine optimum alpha={fine[j]:+.3f} delta={vals[j]:+.3f}")
    results.append(dict(emb_wd=emb_wd, c_innov=c_innov, best_delta=vals[j], best_alpha=fine[j]))

print()
print("=" * 78)
for x in results:
    print(f"emb_wd={x['emb_wd']:<8} corr(진짜잔차)={x['c_innov']:.4f}  "
          f"최적alpha={x['best_alpha']:+.3f}  최적델타={x['best_delta']:+.3f}")
log(f"총 {time.time()-t0:.0f}s")
