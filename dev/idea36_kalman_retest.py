"""March Mania 리서치에서 나온 '팀(선수) latent 실력 상태공간 추정' 아이디어 재검증.
phase20에서 이미 시도됐던 kalman_ability(theta_s=theta_{s-1}+drift, 이항관측 갱신)가
당시 67피처(v7c) 기준 fold delta +10.7(전체교체)을 보였으나 그 이후 162피처(v27/28) 아키텍처로
넘어오며 kal_post가 누락되고(crosses.py의 x_kal_minus_career조차 지금은 사실상
inseason_success_smooth 기반으로 fallback) 프로덕션에 재도입되지 않았다.
현재 162피처 기준 fold A에서 kalman 4피처를 추가했을 때 여전히 신호가 있는지 확인.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from inseason import build_season_end_table
from kalman_ability import (build_kalman_table, estimate_process_noise, transform_kalman)

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()

upto, val, tag = 2023, 2024, "A"
tr_m = seasons <= upto
va_m = seasons == val
yv = y[va_m]
r = yv.mean(); BS = r * (1 - r)
w = 0.5 ** ((upto - seasons) / 2.0)


def sc(p):
    return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)


log("칼만 실력 테이블 구성 (train<=2023만 사용, leakage 안전)...")
df_tr = meta.loc[tr_m].copy()
sr_all = sorted(meta["season"].unique().tolist())
g_rate = float(y[tr_m].mean())

q = estimate_process_noise(df_tr, entity="pitcher_id")
log(f"  추정 드리프트분산 q={q:.6f}")
theta_tbl, P_tbl = build_kalman_table(df_tr, sr_all, q, g_rate, entity="pitcher_id")

log("in-season 원시 n/rate 계산 (train<=2023으로 만든 테이블만 사용)...")
se = build_season_end_table(df_tr)

full_df = pd.DataFrame({"pitcher_id": pid, "season": seasons})
piv = se.set_index(["pitcher_id", "season"])
idx = pd.MultiIndex.from_arrays([pid, seasons - 1])
N_end = pd.Series(piv["N_end"].reindex(idx).to_numpy(), index=full_df.index).fillna(0).to_numpy(np.float64)
S_end = pd.Series(piv["S_end"].reindex(idx).to_numpy(), index=full_df.index).fillna(0).to_numpy(np.float64)

asof_n = meta["asof_pitcher_n"].to_numpy(np.float64)
asof_rate = meta["asof_pitcher_success_rate"].to_numpy(np.float64)
n_now = np.nan_to_num(asof_n, nan=0.0)
s_now = np.round(np.nan_to_num(asof_rate, nan=0.0) * n_now)
n_season = np.clip(n_now - N_end, 0, None)
s_season = np.clip(s_now - S_end, 0, None)
rate_season = np.divide(s_season, n_season, out=np.full(len(n_season), np.nan), where=n_season > 0)

kal = transform_kalman(full_df, theta_tbl, P_tbl, g_rate, entity="pitcher_id",
                       inseason_n=n_season, inseason_rate=rate_season)
log(f"  kal_post corr with inseason_success_smooth = "
   f"{np.corrcoef(kal['kal_post'], X['inseason_success_smooth'])[0,1]:.4f}")

HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=400, learning_rate=0.05, l2_regularization=5.0,
          early_stopping=True, validation_fraction=0.08, n_iter_no_change=20, random_state=42)

log("baseline 학습...")
ts = time.time()
m0 = HistGradientBoostingClassifier(**HGB)
m0.fit(X.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
p0 = m0.predict_proba(X.loc[va_m])[:, 1]
log(f"  baseline score={sc(p0):.2f} iters={m0.n_iter_} ({time.time()-ts:.0f}s)")

log("+kalman4 (추가) 학습...")
X2 = pd.concat([X, kal], axis=1)
ts = time.time()
m1 = HistGradientBoostingClassifier(**HGB)
m1.fit(X2.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
p1 = m1.predict_proba(X2.loc[va_m])[:, 1]
log(f"  +kalman4 score={sc(p1):.2f}  delta={sc(p1)-sc(p0):+.2f} iters={m1.n_iter_} ({time.time()-ts:.0f}s)")

log("교체: inseason_success_smooth -> kal_post 학습...")
X3 = X.copy()
X3["inseason_success_smooth"] = kal["kal_post"].to_numpy()
ts = time.time()
m2 = HistGradientBoostingClassifier(**HGB)
m2.fit(X3.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
p2 = m2.predict_proba(X3.loc[va_m])[:, 1]
log(f"  교체 score={sc(p2):.2f}  delta={sc(p2)-sc(p0):+.2f} iters={m2.n_iter_} ({time.time()-ts:.0f}s)")

log(f"총 {time.time()-t0:.0f}s")
