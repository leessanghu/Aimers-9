"""idea65 — MLP가 y를 직접 맞히는 대신 v66의 오차(residual = y - v66)를 직접 맞히게.

idea61/62/63은 MLP를 "y를 독립적으로 맞히도록" 학습한 뒤, 부산물로 v66과 오차상관이
낮기를 바라는 구조였다(간접). 이 스크립트는 목적함수 자체를 residual = y - v66로
바꿔 "v66이 못 맞춘 부분을 맞혀라"를 직접 학습시킨다(직접) -- v72(residcorr)와 같은
프레임이지만 입력이 78개 GBDT식 피처가 아니라 162피처+pitcher/batter 임베딩이라는
점이 다르다.

결합은 additive correction: p_final = v66 + alpha*resid_hat (가중평균 아님).
판정은 여전히 로컬 blend 델타가 아니라 rho(오차상관) vs sqrt(B1/B2) 임계식.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 4))
CD = "idea65_cache"
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

# train 구간(<=2023)의 v66 프록시가 필요하다 -- 검증용 fold A는 valid만 있으므로
# train 구간엔 v47 4멤버(base/hurdle/multires/ordinal)의 OOF/과거값이 없다.
# 대신 v66 자체가 아니라 "투수시즌 LOO실력"을 프록시 잔차 타깃으로 쓰면 train 전체에서
# 안전하게 계산 가능하다 (idea58/v72와 동일 정당화: 최소 in-sample 근사).
# 여기서는 실전과 동일하게 아래 근사를 쓴다:
#   train 구간 잔차타깃 = y - 투수시즌LOO실력(비축소, n>=20)
#   valid 구간 성능평가는 진짜 v66(캐시)로 한다 -> 정확한 alpha grid 산출 가능
d = pd.DataFrame({"pid": pid, "season": season, "y": y})
ps = d.groupby(["pid", "season"])["y"].agg(s="sum", n="count")
d = d.join(ps, on=["pid", "season"])
n_arr = d["n"].to_numpy(np.float64)
ability_loo = (d["s"].to_numpy(np.float64) - y) / np.maximum(n_arr - 1, 1)
resid_target = (y - ability_loo).astype(np.float32)
fit_ok = n_arr >= 20
log(f"잔차타깃(투수LOO기준) 커버리지={fit_ok.mean():.2%}  train내 mean={resid_target[tr&fit_ok].mean():+.5f} sd={resid_target[tr&fit_ok].std():.4f}")

vc_p = pd.Series(pid[tr]).value_counts()
vc_b = pd.Series(bid[tr]).value_counts()
pmap = {v: i + 1 for i, v in enumerate(vc_p[vc_p >= MIN_COUNT].index)}
bmap = {v: i + 1 for i, v in enumerate(vc_b[vc_b >= MIN_COUNT].index)}
ip = np.array([pmap.get(v, 0) for v in pid], dtype=np.int64)
ib = np.array([bmap.get(v, 0) for v in bid], dtype=np.int64)
mu, sd_ = Xn[tr].mean(0), Xn[tr].std(0)
sd_[sd_ < 1e-8] = 1.0
Z = np.clip((Xn - mu) / sd_, -10, 10)

fit_mask = tr & fit_ok
fit_idx = np.where(fit_mask)[0]
rng = np.random.default_rng(0)
perm0 = rng.permutation(len(fit_idx))
cut = int(len(fit_idx) * 0.92)
fit_i, es_i = fit_idx[perm0[:cut]], fit_idx[perm0[cut:]]
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


def train_one(seed, emb_wd=3e-2, lr=3e-4, max_ep=40, pat=6):
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
    log(f"  [seed={seed}] 학습완료 best_es={best:.6f} epochs={e+1}")
    net.load_state_dict(best_state)
    return net


log("MLP 학습 (target=y-투수LOO실력, MSE, grad-clip + emb_wd=3e-2)... 2시드로 안정성도 같이 확인")


def predict(net):
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(va_i), 65536):
            j = va_i[s:s + 65536]
            out.append(net(T(Z, j), T(ip, j), T(ib, j)).numpy())
    return np.concatenate(out).astype(np.float64)


rhats = []
for seed in (42, 7):
    net = train_one(seed)
    r_ = predict(net)
    c_innov_s = np.corrcoef(r_, e1 * -1)[0, 1]
    log(f"    [seed={seed}] sd={r_.std():.5f} corr(진짜잔차)={c_innov_s:.4f}")
    rhats.append(r_)
rhat = np.mean(rhats, axis=0)
np.save(f"{CD}/A_residhat.npy", rhat)

c_v66 = np.corrcoef(rhat, v66_va)[0, 1]
c_innov = np.corrcoef(rhat, e1 * -1)[0, 1]  # -e1 = y - v66 = 진짜 잔차
log(f"2시드평균 rhat: mean={rhat.mean():+.5f} sd={rhat.std():.5f}  corr(v66예측)={c_v66:.4f}  corr(진짜잔차)={c_innov:.4f}")

print()
print("alpha grid: p = v66 + alpha*rhat")
rows = []
for a in (-0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5):
    d = sc(v66_va + a * rhat) - B66
    rows.append((a, d))
    print(f"  alpha={a:+.2f}: delta={d:+.3f}")
fine = np.linspace(-0.2, 0.5, 141)
vals = [sc(v66_va + a * rhat) - B66 for a in fine]
j = int(np.argmax(vals))
print(f"  fine optimum alpha={fine[j]:+.3f} delta={vals[j]:+.3f}")
print()
print(f"corr(rhat, 진짜잔차 y-v66) = {c_innov:.5f}  (이게 양수&클수록 residual을 잘 예측한다는 뜻)")
log(f"총 {time.time()-t0:.0f}s")
