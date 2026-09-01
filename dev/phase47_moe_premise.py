"""Gated MoE의 전제 검증 — 트리가 이미 n(표본수)에 따라 신뢰도를 조절하고 있는가?

MoE 제안의 전제: "트리는 표본수에 따른 in-season 신뢰도 조절을 잘 못한다"
그런데 v15는 이미 inseason_n(로그 표본수)을 피처로 갖고 있다.
CatBoost가 이미 조절 중이면 MoE는 같은 일을 더 복잡하게 하는 것뿐이다.

측정 (학습 1회로 전부):
  1) n 구간별 Brier / BSS -> 어느 구간이 유독 나쁜가
  2) n 구간별 '예측이 in-season rate를 얼마나 따라가는가'(회귀 기울기)
     -> n이 작을 때 기울기가 낮으면(=prior 쪽으로 축소) 이미 gate 역할을 하는 것
  3) 구간별 잔차가 in-season rate / prior와 상관이 남는가
     -> 남으면 MoE가 고칠 여지, 없으면 이미 최적
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import K_SMOOTH, build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())
se = build_season_end_table(df)
dins = transform_inseason(df, se, g, sr)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
gr = build_global_rates(df)
dly = transform_lastyear(df, build_lastyear_table(df), gr, sr, k=30.0)

# in-season 원시 관측값 복원 (스무딩 역산)
n_season = np.expm1(dins["inseason_n"].to_numpy(np.float64))
sm = dins["inseason_success_smooth"].to_numpy(np.float64)
raw_cur = np.clip(np.where(n_season > 0,
                           (sm * (n_season + K_SMOOTH) - K_SMOOTH * pp) / np.maximum(n_season, 1e-9),
                           np.nan), 0, 1)
print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
ytr, yva = fold["y_train"], fold["y_valid"]
tr, va = df[df.season <= 2023].index, df[df.season == 2024].index
ti, ei = time_split_es(len(tr))


def stack(i, bf):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


Xtr, Xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                   l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                        verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
p = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
print(f"v15 구성 score={max(0, evaluate(yva, p)['bss']*1e5):.1f}  ({time.time()-t0:.0f}s)\n", flush=True)

nv = n_season[va]
rv = raw_cur[va]
prv = pp[va]
e = yva - p
r_all = yva.mean()

bins = [(0, 1, "n=0 (첫등장)"), (1, 30, "1<=n<30"), (30, 100, "30<=n<100"),
        (100, 300, "100<=n<300"), (300, 1e9, "n>=300")]

print("=" * 92, flush=True)
print("구간별 성능 + 예측이 in-season rate를 따라가는 정도(기울기)", flush=True)
print("=" * 92, flush=True)
print(f"{'구간':16s}{'n행':>9s}{'BSS':>9s}{'점수':>9s}{'pred~raw기울기':>15s}{'pred~prior기울기':>17s}", flush=True)
for lo, hi, nm in bins:
    m = (nv >= lo) & (nv < hi)
    if m.sum() < 500:
        continue
    ym, pm = yva[m], p[m]
    rm = np.clip(np.nan_to_num(rv[m], nan=g), 0, 1)
    pm_ = prv[m]
    bss = 1 - np.mean((pm - ym) ** 2) / (ym.mean() * (1 - ym.mean()))
    # 예측을 raw in-season rate와 prior로 회귀 -> 어느 쪽을 얼마나 따라가나
    A = np.column_stack([np.ones(m.sum()), rm, pm_])
    beta = np.linalg.lstsq(A, pm, rcond=None)[0]
    print(f"{nm:16s}{m.sum():>9,}{bss:>9.5f}{max(0,bss*1e5):>9.1f}{beta[1]:>15.4f}{beta[2]:>17.4f}", flush=True)

print("\n" + "=" * 92, flush=True)
print("구간별 잔차 상관 (0에 가까울수록 이미 최적 -> MoE 여지 없음)", flush=True)
print("=" * 92, flush=True)
print(f"{'구간':16s}{'corr(e, raw)':>16s}{'corr(e, prior)':>17s}{'corr(e, log n)':>17s}", flush=True)
for lo, hi, nm in bins:
    m = (nv >= lo) & (nv < hi)
    if m.sum() < 500:
        continue
    em = e[m]
    rm = np.clip(np.nan_to_num(rv[m], nan=g), 0, 1)
    c1 = np.corrcoef(em, rm)[0, 1] if rm.std() > 1e-9 else 0.0
    c2 = np.corrcoef(em, prv[m])[0, 1] if prv[m].std() > 1e-9 else 0.0
    c3 = np.corrcoef(em, np.log1p(nv[m]))[0, 1] if nv[m].std() > 1e-9 else 0.0
    print(f"{nm:16s}{c1:>16.5f}{c2:>17.5f}{c3:>17.5f}", flush=True)

print("\n[해석] 기울기가 n 증가에 따라 raw쪽으로 올라가고 prior쪽으로 내려가면", flush=True)
print("       트리가 이미 gate 역할을 하고 있다는 뜻 -> MoE 추가 이득 없음", flush=True)
print(f"\n총 {time.time()-t0:.0f}s", flush=True)
