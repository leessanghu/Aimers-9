"""idea66 — MLP를 "강하게"가 아니라 "v66과 덜 닮게" 직접 학습시킨다.

수학적 사실(idea62/63에서 확정): blend 이득 조건은 rho(오차상관) < sqrt(B1/B2).
지금 미달폭은 rho=0.999052 vs 임계 0.998929, 차이 0.000123뿐이다. 이걸 채우려면
MLP 단독성능을 711->737(+26점) 올리거나, **rho를 0.000123만 낮추면** 된다. 후자가
훨씬 작은 목표다.

방법: loss에 상관 페널티를 추가한다.
    loss = MSE(sigmoid(out), y) + lambda * corr(sigmoid(out), ability_proxy)^2
ability_proxy는 train 전 구간에서 안전하게 계산 가능한, PC1(투수실력)과 상관이
높은 기존 피처 3개(x_ability_here, inseason_success_smooth, inseason_cmd_index)의
표준화 평균이다. 실제 v66 예측은 train 구간에 존재하지 않으므로(valid만 캐시됨)
직접 쓸 수 없고, 이 근사 프록시로 대신한다.

판정은 **진짜 v66 오차**(fold A valid 캐시)에 대한 rho로 한다(프록시가 아니라).
lambda를 0(대조군)부터 스윕해 성능-상관 트레이드오프 곡선을 그리고, 손익분기
부등식을 만족하는 지점이 있는지 확인한다.

안정성 조치(idea63/65에서 확인된 문제 반영): gradient clipping, 임베딩
weight_decay 3e-2로 강화.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 6))
CD = "idea66_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
MIN_COUNT = 30
EMB = 8
LAMBDAS = [0.0, 0.002, 0.005, 0.01, 0.02]


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
v66_va = (.1824 * base + .2432 * hur + .0608 * mr + .1216 * od + .1520 * mo
          + .08 * cb + .08 * cr + .08 * f5)
B66 = sc(v66_va)
e1 = v66_va - yv
B1 = float(np.mean(e1 ** 2))
log(f"v66local={B66:.2f}  B1={B1:.6f}  손익분기 부등식: rho < sqrt(B1/B2)")

# ability_proxy: train 전구간에서 안전한 PC1근사(v66 자체는 train행에 없음)
proxy_cols = ["x_ability_here", "inseason_success_smooth", "inseason_cmd_index"]
Pxy = X[proxy_cols].to_numpy(np.float64)
Pxy = (Pxy - Pxy[tr].mean(0)) / Pxy[tr].std(0)
ability_proxy = Pxy.mean(1).astype(np.float32)
log(f"ability_proxy(train) corr(진짜v66오차, valid에서 검증용) 계산 예정")

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


def batch_corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)


def run(lam, seed=42, max_ep=40, pat=6, lr=5e-4):
    log(f"--- lambda={lam} ---")
    torch.manual_seed(seed)
    net = Net(Xn.shape[1], len(pmap) + 1, len(bmap) + 1)
    opt = torch.optim.AdamW([
        {"params": list(net.mlp.parameters()), "weight_decay": 1e-5},
        {"params": list(net.ep.parameters()) + list(net.eb.parameters()), "weight_decay": 3e-2},
    ], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    apx = torch.as_tensor(ability_proxy)
    ez, ep_, eb_, ey, ew, eapx = T(Z, es_i), T(ip, es_i), T(ib, es_i), T(y, es_i), T(w_rec, es_i), apx[es_i]
    best, best_state, bad = np.inf, None, 0
    for e in range(max_ep):
        net.train()
        perm = np.random.default_rng(seed * 1000 + e).permutation(len(fit_i))
        for s in range(0, len(perm), 8192):
            j = fit_i[perm[s:s + 8192]]
            opt.zero_grad()
            p = torch.sigmoid(net(T(Z, j), T(ip, j), T(ib, j)))
            w = T(w_rec, j)
            mse = ((p - T(y, j)) ** 2 * w).sum() / w.sum()
            pen = batch_corr(p, apx[j]) ** 2 if lam > 0 else torch.zeros(())
            loss = mse + lam * pen
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(ez, ep_, eb_))
            vl = float(((pv - ey) ** 2 * ew).sum() / ew.sum())
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
    np.save(f"{CD}/A_lam{lam}.npy", p)

    Vp = np.mean((p - r) ** 2)
    k = np.mean((yv - r) * (p - r)) / Vp
    pc = r + k * (p - r)
    B2 = float(np.mean((pc - yv) ** 2))
    rho = float(np.corrcoef(e1, pc - yv)[0, 1])
    thr = np.sqrt(B1 / B2)
    bestw, bestd = 0.0, 0.0
    for w in np.linspace(0, 0.4, 161):
        d = sc((1 - w) * v66_va + w * pc) - B66
        if d > bestd:
            bestw, bestd = w, d
    log(f"  단독={sc(p):.2f} 캘리브후={sc(pc):.2f} k={k:.3f} rho={rho:.6f} 임계={thr:.6f} "
        f"여유={thr-rho:+.6f} | 최적w={bestw:.3f} {bestd:+.2f}")
    return dict(lam=lam, solo=sc(p), cal=sc(pc), rho=rho, thr=thr, w=bestw, d=bestd)


results = [run(lam) for lam in LAMBDAS]
print()
print("=" * 100)
print(f"{'lambda':>8s} {'단독':>9s} {'캘리브후':>9s} {'rho':>10s} {'임계':>10s} {'여유':>10s} {'최적w':>7s} {'로컬델타':>9s}")
for x in results:
    print(f"{x['lam']:8.3f} {x['solo']:9.2f} {x['cal']:9.2f} {x['rho']:10.6f} {x['thr']:10.6f} "
          f"{x['thr']-x['rho']:+10.6f} {x['w']:7.3f} {x['d']:+9.2f}")
print()
best = max(results, key=lambda x: x["d"])
print(f"최고: lambda={best['lam']}  캘리브후={best['cal']:.2f}  여유={best['thr']-best['rho']:+.6f}  "
      f"{'>>> 이득권 진입' if best['thr']>best['rho'] else '아직 미달'}")
log(f"총 {time.time()-t0:.0f}s")
