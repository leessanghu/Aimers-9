"""사후 확률보정(calibration) 검증 — team_te=expanding(v23 기준) 위에서, 고정 0.5:0.5
블렌딩 결과를 isotonic regression으로 재보정했을 때 실제 개선이 있는지 확인.

주의: 스태킹(phase57)이 실패한 이유는 held-out(2023년 말)으로 찾은 최적값이 진짜 미래
(2024)엔 안 맞아서였다. Calibration도 같은 holdout에 의존하므로 같은 함정에 빠질 수 있다.
다만 calibration은 두 상관된 모델의 미세한 차이를 짜내는 게 아니라 '예측확률 자체가
정확한가'라는 더 매끄럽고 안정적인 성질을 고치는 거라, 시간에 덜 민감할 수 있다는 가설을
직접 검증한다.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
SEEDS = [42, 7, 10]

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
tr, va = df[df.season <= 2023].index, df[df.season == 2024].index
print(f"베이스 피처 준비 완료 ({time.time()-t0:.0f}s)", flush=True)


def stack(i, bf):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


for seed in SEEDS:
    fold = build_fold(df, 2023, 2024, extra_features=None, seed=seed, include_team_te=True,
                      team_te_mode="expanding")
    ytr, yva = fold["y_train"], fold["y_valid"]
    Xtr, Xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
    ti, ei = time_split_es(len(Xtr))

    h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                       l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=20, random_state=seed).fit(Xtr, ytr)
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            random_seed=seed, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))

    p_blend_va = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
    p_blend_ho = 0.5 * h.predict_proba(Xtr.iloc[ei])[:, 1] + 0.5 * cb.predict_proba(Xtr.iloc[ei])[:, 1]
    y_ho = ytr[ei]

    s_uncal = max(0, evaluate(yva, p_blend_va)["bss"] * 1e5)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6).fit(p_blend_ho, y_ho)
    p_cal = iso.predict(p_blend_va)
    s_cal = max(0, evaluate(yva, p_cal)["bss"] * 1e5)

    calib_gap_before = float(p_blend_va.mean() - yva.mean())
    calib_gap_after = float(p_cal.mean() - yva.mean())

    print(f"seed={seed:4d}  uncalibrated={s_uncal:.1f}  isotonic={s_cal:.1f}  delta={s_cal-s_uncal:+.1f}  "
          f"calib_gap(pred-actual): {calib_gap_before:+.4f} -> {calib_gap_after:+.4f}  "
          f"({time.time()-t0:.0f}s)", flush=True)

print(f"\n총 {time.time()-t0:.0f}s", flush=True)
