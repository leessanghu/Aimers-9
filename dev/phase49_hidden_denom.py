"""hidden denominator workload 7개 — v18 실제 잔차 검증 + 직접 폴드 재확인.

중요한 전례: 이 세션 초반 pitchcount_recover.py로 만든 '거의 같은 메커니즘'
(prev1/3/5_game success+middle rate 쌍으로 분모 복원)의 workload/form 피처를
실제 제출했을 때 -4.2였다. 다운스트림 구성은 다르지만 원천 데이터/트릭은 동일.

다변량 사영은 v17에서 부호반전(+5.6예측->-10.1실제)을 냈으므로 개별 피처만 신뢰.
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
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]


def infer_min_denominator(success_rate, middle_rate, max_q, chunk=4000, eps=5.1e-7):
    q = np.arange(1, max_q + 1, dtype=np.float64)
    s_all = pd.Series(success_rate).to_numpy(dtype=np.float64)
    m_all = pd.Series(middle_rate).to_numpy(dtype=np.float64)
    inferred = np.ones(len(s_all), dtype=np.float64)
    for start in range(0, len(s_all), chunk):
        s = s_all[start:start + chunk, None]
        m = m_all[start:start + chunk, None]
        missing = np.isnan(s[:, 0]) | np.isnan(m[:, 0])
        s = np.nan_to_num(s, nan=0.0)
        m = np.nan_to_num(m, nan=0.0)
        err = np.maximum(np.abs(s * q - np.rint(s * q)), np.abs(m * q - np.rint(m * q))) / q
        valid = err <= eps
        vals = np.where(valid.any(axis=1), valid.argmax(axis=1) + 1, err.argmin(axis=1) + 1)
        vals[missing] = 1.0
        inferred[start:start + len(vals)] = vals
    return inferred


def hidden_denominator_features(df):
    out = pd.DataFrame(index=df.index)
    for k, max_q in ((1, 160), (3, 480), (5, 800)):
        out[f"prev{k}_hidden_total_n"] = infer_min_denominator(
            df[f"asof_pitcher_prev{k}_game_success_rate"],
            df[f"asof_pitcher_prev{k}_game_middle_rate"], max_q)
    out["prev3_hidden_avg_n"] = out["prev3_hidden_total_n"] / 3.0
    out["prev5_hidden_avg_n"] = out["prev5_hidden_total_n"] / 5.0
    out["prev1_vs_prev3_workload"] = out["prev1_hidden_total_n"] - out["prev3_hidden_avg_n"]
    out["prev3_vs_prev5_workload"] = out["prev3_hidden_avg_n"] - out["prev5_hidden_avg_n"]
    return out.astype(np.float64)


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

print("hidden denominator 피처 계산...", flush=True)
dden = hidden_denominator_features(df)
print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)
for c in dden.columns:
    print(f"  {c:26s} SD={dden[c].std():10.3f}  mean={dden[c].mean():8.3f}", flush=True)

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
base = max(0, evaluate(yva, p)["bss"] * 1e5)
print(f"\nv15(=v18기반) 단일시드 구성 score={base:.1f}  ({time.time()-t0:.0f}s)", flush=True)

e = yva - p
r = yva.mean()
bv = r * (1 - r)
Xn = Xva.to_numpy(np.float64)
Xd = np.column_stack([np.ones(len(Xn)), Xn, p, p ** 2])
XtX = Xd.T @ Xd + 1e-3 * np.eye(Xd.shape[1])

print("\n=== hidden denominator 개별 잔차가치 ===", flush=True)
tot = 0.0
for c in dden.columns:
    z = dden.loc[va, c].to_numpy(np.float64)
    if z.std() < 1e-12:
        continue
    zp = z - Xd @ np.linalg.solve(XtX, Xd.T @ z)
    v = zp.var()
    if v < 1e-14:
        print(f"  {c:26s} (전부 설명됨)", flush=True)
        continue
    gain = (np.cov(e, zp)[0, 1] ** 2 / v) / bv * 1e5
    tot += gain
    print(f"  {c:26s} corr(e,z_perp)={np.corrcoef(e, zp)[0,1]:+.5f}  개별={gain:+6.2f}", flush=True)
print(f"  개별 합산(참고) = {tot:+.1f}", flush=True)

print("\n=== 직접 폴드 재확인 (v15구성 + hidden denom 7개, HGB+cat단일) ===", flush=True)
Xtr2 = pd.concat([Xtr, dden.loc[tr].reset_index(drop=True)], axis=1)
Xva2 = pd.concat([Xva, dden.loc[va].reset_index(drop=True)], axis=1)
h2 = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                    l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                    n_iter_no_change=20, random_state=SEED).fit(Xtr2, ytr)
cb2 = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                         verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
cb2.fit(Xtr2.iloc[ti], ytr[ti], eval_set=(Xtr2.iloc[ei], ytr[ei]))
p2 = 0.5 * h2.predict_proba(Xva2)[:, 1] + 0.5 * cb2.predict_proba(Xva2)[:, 1]
s2 = max(0, evaluate(yva, p2)["bss"] * 1e5)
print(f"  baseline={base:.1f}  +hidden_denom7={s2:.1f}  delta={s2-base:+.1f}  (보고받은 값: +9.23, 812.88->822.11)", flush=True)
print(f"\n총 {time.time()-t0:.0f}s", flush=True)
