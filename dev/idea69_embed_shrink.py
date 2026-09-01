"""idea69 — idea64 재시도: 임베딩에 베이지안 축소(shrinkage)를 걸고 GBDT 피처로.

idea64 실패(-145점) 원인: MLP 임베딩은 gradient descent로 train을 그대로 외운
비축소(un-shrunk) 표현이라, HGB가 이걸 "축소 안 된 투수 정체성 치트키"로 써서
과적합했다. 기존 코드베이스의 target encoding들(batterform K=30, count_split
K=880, platoon K=520, inning K=570)은 전부 표본수 기반 베이지안 축소를 쓴다 --
임베딩에도 같은 원리를 적용한다:

    emb_shrunk[id] = emb[id] * n_id / (n_id + K)

n_id=0(unknown, train<30회)이면 완전히 0벡터로 축소되고, n_id가 크면 원본에
가까워진다. K를 스윕해 최적 축소강도를 찾는다.

MLP 재학습은 생략(idea64_cache의 emb_p.npy/emb_b.npy 재사용, ~230s 절약). pmap/bmap은
동일 규칙(MIN_COUNT=30, train<=2023)으로 재구성하면 100% 동일하게 재현된다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

t0 = time.time()
MIN_COUNT = 30
EMB = 8
K_LIST = [3000, 10000, 30000, 100000]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
season = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
bid = meta["batter_id"].to_numpy()

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
log(f"v66local={B66:.2f}")

vc_p = pd.Series(pid[tr]).value_counts()
vc_b = pd.Series(bid[tr]).value_counts()
pmap = {v: i + 1 for i, v in enumerate(vc_p[vc_p >= MIN_COUNT].index)}
bmap = {v: i + 1 for i, v in enumerate(vc_b[vc_b >= MIN_COUNT].index)}
ip = np.array([pmap.get(v, 0) for v in pid], dtype=np.int64)
ib = np.array([bmap.get(v, 0) for v in bid], dtype=np.int64)

# idx별 표본수: idx=0(unknown)은 n=0 -> shrink factor 0
n_p = np.zeros(len(pmap) + 1)
for v, i in pmap.items():
    n_p[i] = vc_p[v]
n_b = np.zeros(len(bmap) + 1)
for v, i in bmap.items():
    n_b[i] = vc_b[v]
n_p_row = n_p[ip]
n_b_row = n_b[ib]
log(f"pitcher vocab={len(pmap)+1}(unknown포함) batter vocab={len(bmap)+1}  "
    f"n_p median(known)={np.median(n_p[n_p>0]):.0f}  n_b median(known)={np.median(n_b[n_b>0]):.0f}")

emb_p = np.load("idea64_cache/emb_p.npy")   # (n_p+1, EMB), idea64에서 학습된 것 재사용
emb_b = np.load("idea64_cache/emb_b.npy")
assert emb_p.shape[0] == len(pmap) + 1 and emb_b.shape[0] == len(bmap) + 1, "vocab 재현 불일치"
Ep_raw = emb_p[ip]
Eb_raw = emb_b[ib]

w_rec = (0.5 ** ((UPTO - season) / 2.0))

HGB = dict(loss="squared_error", max_iter=350, learning_rate=0.03, max_depth=6,
           max_leaf_nodes=31, l2_regularization=10.0, early_stopping=True,
           validation_fraction=0.10, n_iter_no_change=25, random_state=42)


def fit_eval(Xall, label):
    ts = time.time()
    m = HistGradientBoostingRegressor(**HGB)
    m.fit(Xall.loc[tr], y[tr], sample_weight=w_rec[tr])
    p = np.clip(m.predict(Xall.loc[va]), 0, 1)
    log(f"  [{label}] n_iter={m.n_iter_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    return sc(p)


BASELINE = 855.03  # idea64에서 이미 측정(재사용, 재학습 생략)
log(f"baseline(162피처) 참고값(idea64 재사용) = {BASELINE:.2f}")

results = [dict(K="raw(무축소,idea64)", score=709.62, delta=709.62 - BASELINE)]
for K in K_LIST:
    fac_p = n_p_row / (n_p_row + K)
    fac_b = n_b_row / (n_b_row + K)
    Ep = Ep_raw * fac_p[:, None]
    Eb = Eb_raw * fac_b[:, None]
    cols = [f"emb_p{i}" for i in range(EMB)] + [f"emb_b{i}" for i in range(EMB)]
    X_emb = pd.DataFrame(np.concatenate([Ep, Eb], axis=1), columns=cols, index=X.index)
    X_aug = pd.concat([X, X_emb], axis=1)
    s = fit_eval(X_aug, f"K={K}")
    results.append(dict(K=K, score=s, delta=s - BASELINE))

print()
print("=" * 66)
print(f"baseline(162피처, 재사용값) = {BASELINE:.2f}")
print(f"{'K(축소강도)':>16s} {'단독':>9s} {'델타':>9s}")
for x in results:
    print(f"{str(x['K']):>16s} {x['score']:9.2f} {x['delta']:+9.2f}")
best = max(results, key=lambda x: x["delta"])
print(f"\n최고: K={best['K']}  델타={best['delta']:+.2f}  {'>>> 이득' if best['delta']>0 else '여전히 손해'}")
log(f"총 {time.time()-t0:.0f}s")
