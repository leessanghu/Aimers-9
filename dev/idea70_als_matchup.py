"""idea70 — pitcher x batter 매치업 잔차를 ALS(정규화 내장 행렬분해)로 분해.

idea64/69에서 배운 교훈 2가지를 설계에 직접 반영한다:
  1) MLP 임베딩은 gradient descent로 축소 없이 train을 외워 GBDT에 노이즈로
     작용했다(K=100000까지 축소해도 -80대 플라토). ALS는 손실함수 자체에
     L2 정규화가 있어 표본 적은 pitcher/batter는 팩터가 자동으로 0에 가깝게
     눌린다 -- "사후 축소"가 아니라 "학습 자체가 축소된" 설계.
  2) 투수/타자 각각의 순수 실력(주효과)을 남겨두면 팩터가 결국 PC1(실력)을
     재구성해버릴 수 있다(idea61의 concat MLP가 corr(ability)=0.87이었던 것과
     같은 함정). 그래서 분해 대상을 "주효과를 뺀 매치업 잔차"로 한정한다:
         residual[p,b] = mean_y[p,b] - mu_p(축소) - mu_b(축소) + mu_global
     mu_p/mu_b는 batterform.py 관례(K=30)와 동일한 베이지안 축소 사용.

ALS: residual[p,b] ~= dot(u_p, v_b), L2(lambda) 정규화, 관측쌍만 학습(가중치=n_pb).
행별 피처: u_p[pid], v_b[bid], 그리고 스칼라 dot(u_p,v_b)(=ALS의 매치업잔차 예측치
자체, 가장 직접적인 단일 피처). 두 변종을 baseline(162피처)에 각각 추가해 비교:
  A) 스칼라 dot 1개만 추가 (가장 보수적)
  B) u_p+v_b+dot 전체(k*2+1개) 추가
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

t0 = time.time()
K_SHRINK = 30.0   # batterform.py 관례
RANK = 8
LAMBDAS = [1.0, 5.0, 20.0, 50.0]
ALS_ITERS = 15


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
log(f"관측 pitcher x batter 쌍={len(pair):,}  n_pb median={pair['count'].median():.0f}  "
    f"resid mean={pair['resid'].mean():+.5f} std={pair['resid'].std():.4f}")

pitchers = sorted(set(pair.index.get_level_values(0)))
batters = sorted(set(pair.index.get_level_values(1)))
p_idx = {p: i for i, p in enumerate(pitchers)}
b_idx = {b: i for i, b in enumerate(batters)}
rows_p = np.array([p_idx[p] for p, b in pair.index])
rows_b = np.array([b_idx[b] for p, b in pair.index])
resid = pair["resid"].to_numpy(np.float64)
w_pb = np.sqrt(pair["count"].to_numpy(np.float64))  # 과도한 지배 방지로 sqrt 가중


def als(lmbda, rank=RANK, iters=ALS_ITERS, seed=0):
    rng = np.random.default_rng(seed)
    U = rng.normal(0, 0.01, (len(pitchers), rank))
    V = rng.normal(0, 0.01, (len(batters), rank))
    # 피처별 관측 인덱스 미리 그룹화
    from collections import defaultdict
    obs_by_p = defaultdict(list)
    obs_by_b = defaultdict(list)
    for i in range(len(resid)):
        obs_by_p[rows_p[i]].append(i)
        obs_by_b[rows_b[i]].append(i)
    for it in range(iters):
        # U 업데이트 (V 고정)
        for pi, idxs in obs_by_p.items():
            idxs = np.array(idxs)
            Vb = V[rows_b[idxs]]
            w = w_pb[idxs]
            A = (Vb * w[:, None]).T @ Vb + lmbda * np.eye(rank)
            bvec = (Vb * w[:, None]).T @ resid[idxs]
            U[pi] = np.linalg.solve(A, bvec)
        # V 업데이트 (U 고정)
        for bi, idxs in obs_by_b.items():
            idxs = np.array(idxs)
            Up = U[rows_p[idxs]]
            w = w_pb[idxs]
            A = (Up * w[:, None]).T @ Up + lmbda * np.eye(rank)
            bvec = (Up * w[:, None]).T @ resid[idxs]
            V[bi] = np.linalg.solve(A, bvec)
    pred = np.sum(U[rows_p] * V[rows_b], axis=1)
    train_rmse = np.sqrt(np.mean((pred - resid) ** 2))
    return U, V, train_rmse


def row_features(U, V, rank):
    up = np.zeros((len(X), rank))
    vb = np.zeros((len(X), rank))
    for i, p in enumerate(pid):
        j = p_idx.get(p)
        if j is not None:
            up[i] = U[j]
    for i, b in enumerate(bid):
        j = b_idx.get(b)
        if j is not None:
            vb[i] = V[j]
    dot = np.sum(up * vb, axis=1)
    return up, vb, dot


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


log("baseline(162피처) 학습...")
BASELINE = fit_eval(X, "baseline_162")

log(f"lambda 스윕: A안(scalar dot 1개만 추가), rank={RANK}")
results = []
for lam in LAMBDAS:
    ts = time.time()
    U, V, trmse = als(lam)
    up, vb, dot = row_features(U, V, RANK)
    log(f"  [lambda={lam}] ALS완료 train_rmse={trmse:.4f} ({time.time()-ts:.0f}s)")
    Xa = X.copy()
    Xa["als_dot"] = dot
    sA = fit_eval(Xa, f"A(dot only) lam={lam}")
    results.append(dict(variant="A_dot_only", lam=lam, score=sA, delta=sA - BASELINE))

best = max(results, key=lambda x: x["delta"])
log(f"A안 최고: lambda={best['lam']}  델타={best['delta']:+.2f}")

log(f"B안(u_p+v_b+dot 전체 {RANK*2+1}개) 최적lambda={best['lam']}로 시도")
U, V, trmse = als(best["lam"])
up, vb, dot = row_features(U, V, RANK)
cols_u = [f"als_up{i}" for i in range(RANK)]
cols_v = [f"als_vb{i}" for i in range(RANK)]
Xb = pd.concat([X, pd.DataFrame(up, columns=cols_u, index=X.index),
                pd.DataFrame(vb, columns=cols_v, index=X.index)], axis=1)
Xb["als_dot"] = dot
sB = fit_eval(Xb, f"B(full) lam={best['lam']}")
results.append(dict(variant="B_full", lam=best["lam"], score=sB, delta=sB - BASELINE))

print()
print("=" * 66)
print(f"baseline(162피처) = {BASELINE:.2f}")
print(f"{'변종':>14s} {'lambda':>8s} {'단독':>9s} {'델타':>9s}")
for x in results:
    print(f"{x['variant']:>14s} {x['lam']:8.1f} {x['score']:9.2f} {x['delta']:+9.2f}")
best_all = max(results, key=lambda x: x["delta"])
print(f"\n최고: {best_all['variant']} lambda={best_all['lam']}  델타={best_all['delta']:+.2f}  "
      f"{'>>> 이득' if best_all['delta']>0 else '여전히 손해'}")
log(f"총 {time.time()-t0:.0f}s")
