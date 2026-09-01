"""idea64 — MLP 임베딩을 "확률"이 아니라 "피처"로 GBDT에 주입.

idea62/63: MLP를 v66과 선형 blend하려면 오차상관이 sqrt(B1/B2) 미만이어야 하는데
Brier차이가 작아 임계값이 0.999대라 넘기 힘들다(idea61 corr=0.807이지만 단독성능이
약해서 손익분기 735~737점을 못 넘길 수 있음).

이 스크립트는 그 제약을 완전히 피해간다: MLP 확률 자체를 섞는 게 아니라, 학습된
pitcher/batter 16차원 임베딩 벡터를 **HGB의 입력 피처**로 추가한다. GBDT가 이미
잘 캘리브레이션돼 있으므로, "MLP가 좋은 확률을 만드는가"가 아니라 "GBDT가 이 새
피처로 더 나은 split을 찾는가"만 물으면 된다 -> 선형blend 상관임계 자체가 무관해짐.

절차:
  1) fold A train(<=2023)로 MLP(mse_emb8 레시피, embedding dim=8) 학습, 임베딩 추출
  2) 각 행에 pitcher/batter 임베딩(16차원)을 피처로 붙임 (unknown id는 0벡터)
  3) HGB 두 개(baseline 162피처 vs 162+16피처) fold A에서 학습/비교 -> 순수 피처가치
  4) 참고로 corr(v66)과 blend 손익분기도 같이 확인(부수 정보, 판정은 3)이 메인)
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingRegressor

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 4))
CD = "idea64_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
MIN_COUNT = 30
EMB = 8


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
B66 = sc(v66)
log(f"v66local={B66:.2f}")

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


log("MLP 학습 (임베딩 추출용, mse_emb8 레시피 seed=42)...")
torch.manual_seed(42)
net = Net(Xn.shape[1], len(pmap) + 1, len(bmap) + 1)
opt = torch.optim.AdamW([
    {"params": list(net.mlp.parameters()), "weight_decay": 1e-5},
    {"params": list(net.ep.parameters()) + list(net.eb.parameters()), "weight_decay": 1e-3},
], lr=5e-4)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
ez, ep_, eb_, ey, ew = T(Z, es_i), T(ip, es_i), T(ib, es_i), T(y, es_i), T(w_rec, es_i)
best, best_state, bad = np.inf, None, 0
for e in range(40):
    net.train()
    perm = np.random.default_rng(42 + e).permutation(len(fit_i))
    for s in range(0, len(perm), 8192):
        j = fit_i[perm[s:s + 8192]]
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
        if bad >= 6:
            break
log(f"  MLP 학습완료 best_es={best:.6f} epochs={e+1}")
net.load_state_dict(best_state)

emb_p = net.ep.weight.detach().numpy()   # (n_p+1, EMB)
emb_b = net.eb.weight.detach().numpy()
np.save(f"{CD}/emb_p.npy", emb_p)
np.save(f"{CD}/emb_b.npy", emb_b)

Ep = emb_p[ip]   # (N, EMB) 행별 투수 임베딩
Eb = emb_b[ib]
emb_cols = [f"emb_p{i}" for i in range(EMB)] + [f"emb_b{i}" for i in range(EMB)]
X_emb = pd.DataFrame(np.concatenate([Ep, Eb], axis=1), columns=emb_cols, index=X.index)
log(f"임베딩 피처 {X_emb.shape[1]}개 구성 완료")

HGB = dict(loss="squared_error", max_iter=350, learning_rate=0.03, max_depth=6,
           max_leaf_nodes=31, l2_regularization=10.0, early_stopping=True,
           validation_fraction=0.10, n_iter_no_change=25, random_state=42)


def fit_eval(Xall, label):
    ts = time.time()
    m = HistGradientBoostingRegressor(**HGB)
    m.fit(Xall.loc[tr], y[tr].astype(np.float64), sample_weight=w_rec[tr])
    p = np.clip(m.predict(Xall.loc[va]), 0, 1)
    log(f"  [{label}] n_iter={m.n_iter_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    return p, m


log("=== baseline: 162피처만 ===")
p_base, m_base = fit_eval(X, "baseline_162")
log("=== 확장: 162 + 임베딩16 ===")
X_aug = pd.concat([X, X_emb], axis=1)
p_aug, m_aug = fit_eval(X_aug, "aug_178")

print()
print("=" * 78)
print(f"baseline(162피처) 단독 = {sc(p_base):.2f}")
print(f"aug(162+emb16)   단독 = {sc(p_aug):.2f}   델타 = {sc(p_aug)-sc(p_base):+.2f}")
print("=> 이 델타가 양수면 '임베딩 정보가 GBDT 피처로서 실제 가치있다'는 직접 증거.")
print("   (선형blend 상관임계와 무관 -- 단일모델 내 피처 비교이므로 그 제약 자체가 없음)")

c_v66_base = np.corrcoef(p_base, v66)[0, 1]
c_v66_aug = np.corrcoef(p_aug, v66)[0, 1]
print(f"\n참고: corr(v66, baseline)={c_v66_base:.4f}  corr(v66, aug)={c_v66_aug:.4f}")

log(f"총 {time.time()-t0:.0f}s")
