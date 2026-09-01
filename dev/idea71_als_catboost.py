"""idea71 — idea64/69/70의 방법론 구멍 수정: HGB 대신 CatBoost로 재검증.

idea64/69/70은 전부 sklearn HistGradientBoostingRegressor로 "피처 추가가 도움되는가"를
판정했다. 이건 v72(codex residcorr)에서 가져온 패턴을 그대로 재사용한 것인데, 실제
v66 앙상블의 무게중심(hurdle .32/multires .08/ordinal .16/midother .20/condball
countresid/future50 각 .08)은 거의 다 CatBoost다. 잘못된 모델클래스로 판정했을
가능성이 있다 -- HGB의 히스토그램 분할 방식과 CatBoost의 ordered boosting은
"약한 피처 여러 개 추가"에 대한 민감도가 다를 수 있다.

이 스크립트는 idea70(ALS lambda=50, 최선)과 idea69(임베딩 K=30000, 최선)를
CatBoost(전체 프로덕션과 동일 CAT_PARAMS)로 재검증한다.
"""
import os
import sys
import time
from collections import defaultdict

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

t0 = time.time()
K_SHRINK = 30.0
RANK = 8


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
w_rec = (0.5 ** ((UPTO - season) / 2.0))

CAT_PARAMS = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                   random_seed=42, loss_function="RMSE", early_stopping_rounds=50)


def fit_eval(Xall, label):
    ts = time.time()
    n_es = int(tr.sum() * 0.92)
    tr_idx = np.where(tr)[0]
    fit_i, es_i = tr_idx[:n_es], tr_idx[n_es:]
    m = CatBoostRegressor(**CAT_PARAMS)
    m.fit(Xall.iloc[fit_i], y[fit_i], sample_weight=w_rec[fit_i],
          eval_set=(Xall.iloc[es_i], y[es_i]))
    p = np.clip(m.predict(Xall.loc[va]), 0, 1)
    log(f"  [{label}] best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    return sc(p)


log("baseline(162피처, CatBoost) 학습...")
BASELINE = fit_eval(X, "baseline_162_cat")

# ---- idea70 재현: ALS lambda=50, dot 피처 1개 ----
log("ALS(lambda=50) 매치업잔차 재계산...")
g = float(y[tr].mean())
d_tr = pd.DataFrame({"pid": pid[tr], "bid": bid[tr], "y": y[tr]})
vc_p = d_tr.groupby("pid")["y"].agg(["sum", "count"])
mu_p_map = ((vc_p["sum"] + K_SHRINK * g) / (vc_p["count"] + K_SHRINK)).to_dict()
vc_b = d_tr.groupby("bid")["y"].agg(["sum", "count"])
mu_b_map = ((vc_b["sum"] + K_SHRINK * g) / (vc_b["count"] + K_SHRINK)).to_dict()
pair = d_tr.groupby(["pid", "bid"])["y"].agg(["sum", "count"])
pair["mean"] = pair["sum"] / pair["count"]
pair["mu_p"] = pair.index.get_level_values(0).map(mu_p_map)
pair["mu_b"] = pair.index.get_level_values(1).map(mu_b_map)
pair["resid"] = pair["mean"] - pair["mu_p"] - pair["mu_b"] + g
pitchers = sorted(set(pair.index.get_level_values(0)))
batters = sorted(set(pair.index.get_level_values(1)))
p_idx = {p: i for i, p in enumerate(pitchers)}
b_idx = {b: i for i, b in enumerate(batters)}
rows_p = np.array([p_idx[p] for p, b in pair.index])
rows_b = np.array([b_idx[b] for p, b in pair.index])
resid = pair["resid"].to_numpy(np.float64)
w_pb = np.sqrt(pair["count"].to_numpy(np.float64))


def als(lmbda, rank=RANK, iters=15, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.01, (len(pitchers), rank))
    V = rng.normal(0, 0.01, (len(batters), rank))
    obs_by_p, obs_by_b = defaultdict(list), defaultdict(list)
    for i in range(len(resid)):
        obs_by_p[rows_p[i]].append(i)
        obs_by_b[rows_b[i]].append(i)
    for it in range(iters):
        for pi, idxs in obs_by_p.items():
            idxs = np.array(idxs); Vb = V[rows_b[idxs]]; w = w_pb[idxs]
            A = (Vb * w[:, None]).T @ Vb + lmbda * np.eye(rank)
            bvec = (Vb * w[:, None]).T @ resid[idxs]
            U[pi] = np.linalg.solve(A, bvec)
        for bi, idxs in obs_by_b.items():
            idxs = np.array(idxs); Up = U[rows_p[idxs]]; w = w_pb[idxs]
            A = (Up * w[:, None]).T @ Up + lmbda * np.eye(rank)
            bvec = (Up * w[:, None]).T @ resid[idxs]
            V[bi] = np.linalg.solve(A, bvec)
    return U, V


U, V = als(50.0)
up = np.zeros((len(X), RANK))
vb = np.zeros((len(X), RANK))
for i, p in enumerate(pid):
    j = p_idx.get(p)
    if j is not None:
        up[i] = U[j]
for i, b in enumerate(bid):
    j = b_idx.get(b)
    if j is not None:
        vb[i] = V[j]
dot = np.sum(up * vb, axis=1)
log(f"ALS dot 피처 구성완료 (train내 std={dot[tr].std():.5f})")

X_als = X.copy()
X_als["als_dot"] = dot
S_ALS = fit_eval(X_als, "als_dot_cat")

# ---- idea69 재현: 임베딩 K=30000 축소 ----
log("임베딩(K=30000 축소) 피처 재구성...")
MIN_COUNT = 30
vc_p2 = pd.Series(pid[tr]).value_counts()
vc_b2 = pd.Series(bid[tr]).value_counts()
pmap = {v: i + 1 for i, v in enumerate(vc_p2[vc_p2 >= MIN_COUNT].index)}
bmap = {v: i + 1 for i, v in enumerate(vc_b2[vc_b2 >= MIN_COUNT].index)}
ip = np.array([pmap.get(v, 0) for v in pid], dtype=np.int64)
ib = np.array([bmap.get(v, 0) for v in bid], dtype=np.int64)
n_p = np.zeros(len(pmap) + 1)
for v, i in pmap.items():
    n_p[i] = vc_p2[v]
n_b = np.zeros(len(bmap) + 1)
for v, i in bmap.items():
    n_b[i] = vc_b2[v]
emb_p = np.load("idea64_cache/emb_p.npy")
emb_b = np.load("idea64_cache/emb_b.npy")
K = 30000.0
fac_p = n_p[ip] / (n_p[ip] + K)
fac_b = n_b[ib] / (n_b[ib] + K)
Ep = emb_p[ip] * fac_p[:, None]
Eb = emb_b[ib] * fac_b[:, None]
cols = [f"emb_p{i}" for i in range(8)] + [f"emb_b{i}" for i in range(8)]
X_emb = pd.concat([X, pd.DataFrame(np.concatenate([Ep, Eb], axis=1), columns=cols, index=X.index)], axis=1)
S_EMB = fit_eval(X_emb, "emb_shrunk_K30000_cat")

print()
print("=" * 66)
print(f"baseline(CatBoost, 162피처)     = {BASELINE:.2f}")
print(f"+ALS dot(lambda=50)             = {S_ALS:.2f}  델타={S_ALS-BASELINE:+.2f}")
print(f"+임베딩축소(K=30000)             = {S_EMB:.2f}  델타={S_EMB-BASELINE:+.2f}")
log(f"총 {time.time()-t0:.0f}s")
