"""idea63 — MLP를 앙상블 이득 임계선(737점) 위로 올린다.

idea62에서 mse_emb8이 캘리브후 711.20, 오차상관 0.999052, 임계값 0.998929로
여유 -0.000123. 즉 **26점만 더 올리면 blend가 이득으로 전환**한다.

  앙상블 이득 조건:  rho(오차상관) < sqrt(B1/B2)
  B1=v66 Brier 고정 -> MLP Brier B2가 내려가면 임계값이 올라간다.
  B2 = bs*(1-737/1e5) 에서 임계값이 현재 rho와 같아진다 -> 737점이 손익분기.

두 축으로 공략:
  S1 시드 앙상블 — MLP는 분산이 크다(idea61 단독 246 vs idea62 680, 설정 하나로
     434점 변동). 여러 시드를 평균하면 분산성분이 줄어 단조 개선이 기대된다.
  S2 업데이트 횟수 — batch 8192는 epoch당 137스텝뿐이라 1~2 epoch만에 es가 바닥.
     batch를 줄여 스텝을 늘리면 더 깊이 수렴할 수 있다.

캘리브레이션(Cov/Var 축소)은 train에서 추정 가능하므로 프로덕션 이식에 문제없다.
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
CD = "idea63_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
MIN_COUNT = 30
EMB = 8

RUNS = [
    dict(tag="b8192_lr5e4", batch=8192, lr=5e-4, ep=40, pat=6, seeds=[42, 7, 13, 99, 2024]),
    dict(tag="b2048_lr3e4", batch=2048, lr=3e-4, ep=25, pat=5, seeds=[42, 7, 13]),
]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float32)
season = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
bid = meta["batter_id"].to_numpy()
Xn = X.to_numpy(np.float32)

UPTO, VAL = 2023, 2024
tr, va = season <= UPTO, season == VAL
yv = y[va].astype(np.float64)
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
v66 = (.1824 * base + .2432 * hur + .0608 * mr + .1216 * od + .1520 * mo
       + .08 * cb + .08 * cr + .08 * f5)
e1 = v66 - yv
B1 = float(np.mean(e1 ** 2))
B66 = sc(v66)
BREAKEVEN = 1e5 * (1 - (B1 / 0.999052 ** 2) / bs)
log(f"v66local={B66:.2f}  B1={B1:.6f}  손익분기 MLP점수~{BREAKEVEN:.1f}")

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


def train_one(seed, batch, lr, max_ep, pat, tag):
    f = f"{CD}/A_{tag}_s{seed}.npy"
    if os.path.exists(f):
        return np.load(f)
    torch.manual_seed(seed)
    net = Net(Xn.shape[1], len(pmap) + 1, len(bmap) + 1)
    opt = torch.optim.AdamW([
        {"params": list(net.mlp.parameters()), "weight_decay": 1e-5},
        {"params": list(net.ep.parameters()) + list(net.eb.parameters()), "weight_decay": 1e-3},
    ], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    ez, ep_, eb_, ey, ew = T(Z, es_i), T(ip, es_i), T(ib, es_i), T(y, es_i), T(w_rec, es_i)
    best, best_state, bad = np.inf, None, 0
    ts = time.time()
    for e in range(max_ep):
        net.train()
        perm = np.random.default_rng(seed * 1000 + e).permutation(len(fit_i))
        for s in range(0, len(perm), batch):
            j = fit_i[perm[s:s + batch]]
            opt.zero_grad()
            w = T(w_rec, j)
            loss = (((torch.sigmoid(net(T(Z, j), T(ip, j), T(ib, j))) - T(y, j)) ** 2) * w).sum() / w.sum()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float((((torch.sigmoid(net(ez, ep_, eb_)) - ey) ** 2) * ew).sum() / ew.sum())
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    net.load_state_dict(best_state)
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(va_i), 65536):
            j = va_i[s:s + 65536]
            out.append(torch.sigmoid(net(T(Z, j), T(ip, j), T(ib, j))).numpy())
    p = np.concatenate(out).astype(np.float64)
    np.save(f, p)
    log(f"    [{tag}/s{seed}] es={best:.6f} epochs={e+1} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    return p


def evaluate(p, label):
    Vp = np.mean((p - r) ** 2)
    k = np.mean((yv - r) * (p - r)) / Vp
    pc = r + k * (p - r)
    e2 = pc - yv
    B2 = float(np.mean(e2 ** 2))
    rho = float(np.corrcoef(e1, e2)[0, 1])
    thr = np.sqrt(B1 / B2)
    bestw, bestd = 0.0, 0.0
    for w in np.linspace(0, 0.4, 161):
        d = sc((1 - w) * v66 + w * pc) - B66
        if d > bestd:
            bestw, bestd = w, d
    log(f"  {label:22s} 단독={sc(p):7.2f} 캘리브후={sc(pc):7.2f} k={k:.3f} "
        f"rho={rho:.6f} 임계={thr:.6f} 여유={thr-rho:+.6f} | 최적w={bestw:.3f} {bestd:+.2f}")
    return dict(label=label, cal=sc(pc), rho=rho, thr=thr, w=bestw, d=bestd)


results = []
for R in RUNS:
    log(f"=== {R['tag']}: batch={R['batch']} lr={R['lr']} seeds={R['seeds']} ===")
    ps = []
    for s in R["seeds"]:
        p = train_one(s, R["batch"], R["lr"], R["ep"], R["pat"], R["tag"])
        ps.append(p)
        results.append(evaluate(np.mean(ps, axis=0), f"{R['tag']} {len(ps)}seed"))

print()
print("=" * 96)
print(f"손익분기 ~{BREAKEVEN:.1f}점 (여유>0 이면 blend 이득)")
for x in results:
    print(f"  {x['label']:22s} 캘리브후={x['cal']:7.2f} 여유={x['thr']-x['rho']:+.6f} "
          f"최적w={x['w']:.3f} 로컬델타={x['d']:+.2f}  {'>>> 이득' if x['thr']>x['rho'] else '손해'}")
log(f"총 {time.time()-t0:.0f}s")
