"""phase79 — objective(손실함수) 미스매치 검증.

우린 Logloss(HGB classifier의 log_loss, CatBoost Logloss)로 학습하고 Brier(=MSE)로
채점받는다. 로짓 gradient가 다르다:
    Logloss: (p-y)
    Brier:   2(p-y)*p(1-p)
p가 0.1~0.5 범위면 p(1-p)가 2.8배 차이나서 Logloss는 저확률 꼬리를 과대가중한다.
RMSE/squared_error로 직접 0/1 타겟을 회귀하면 Brier와 같은 손실을 최적화하게 된다.

비교 대상 (전부 동일 폴드 train<=2023 -> valid=2024, 동일 v28 162피처, 동일 recency weight):
    HGB  : HistGradientBoostingClassifier(log_loss)  vs  HistGradientBoostingRegressor(squared_error)
    CatBoost(단일시드 42) : Logloss  vs  RMSE
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

import batter_split as bsplit
from batterform import K_BATTER, build_batter_table, transform_batter
from career_volatility import K_VOL, build_volatility_table, transform_volatility
from count_split import K_COUNT, build_count_table, transform_count
from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from formfeat import build_role_table, transform_form, transform_role
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from inseason_full import build_global_priors, build_season_end_table_full, transform_inseason_full
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from phase2_common import time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
SEED = 42
VALID_SEASON = 2024

HGB_CLS_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                      l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=20, random_state=SEED)
HGB_REG_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                      l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=20, random_state=SEED, loss="squared_error")

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


def score(y, p):
    r = y.mean()
    return 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r)))


log("데이터 로드 + 피처 재구성 (v28 162개)...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy()
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())

fb = FeatureBuilder(seed=SEED, include_raw_rates=False, team_te_mode="expanding").fit(df)
X_base = fb.transform_train_oof(df).reset_index(drop=True)
se = build_season_end_table(df)
X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
X_plt = transform_platoon(df, build_platoon_table(df), prior, sr, k=K_PLATOON).reset_index(drop=True)
it, io = build_inning_table(df), build_inning_offset(df)
X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)
X_cnt = transform_count(df, build_count_table(df), prior, sr, k=K_COUNT).reset_index(drop=True)
X_pt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), prior, g, sr).reset_index(drop=True)
X_ly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0).reset_index(drop=True)
X_vol = transform_volatility(df, build_volatility_table(se), sr, k=K_VOL).reset_index(drop=True)
role_tbl = build_role_table(df)
X_role = transform_role(df, role_tbl, sr).reset_index(drop=True)
base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
X_form = transform_form(df, X_role, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                        base_middle).reset_index(drop=True)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
X_tm = transform_trackman(df, prof, sr).reset_index(drop=True)
lown_thr = float(df["asof_pitcher_n"].fillna(0).median())
X_tmx = add_lown_interactions(X_tm, df["asof_pitcher_n"].to_numpy(np.float64), lown_thr).reset_index(drop=True)
X_bat = transform_batter(df, build_batter_table(df), sr, g, k=K_BATTER).reset_index(drop=True)
n_end_row = np.nan_to_num(piv["N_end"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
X_isf = transform_inseason_full(df, build_season_end_table_full(df), build_global_priors(df), sr,
                                n_end_row, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                                X_ins["inseason_reverse_smooth"].to_numpy(np.float64)).reset_index(drop=True)
g_bmid = float(df["asof_batter_middle_rate"].mean(skipna=True))
X_bmid = bsplit.transform_batter_middle(df, bsplit.build_batter_middle_table(df), sr, g_bmid).reset_index(drop=True)
bmarg = bsplit.build_batter_marginal(df)
b_prior = bsplit.lookup_batter_prior(df, bmarg, sr, g)
X_bplat = bsplit.transform_bplatoon(df, bsplit.build_bplatoon_table(df), b_prior, sr,
                                    k=bsplit.K_BPLATOON).reset_index(drop=True)

X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
C = add_crosses(X)
X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm, X_tmx, X_bat,
               X_isf, X_bmid, X_bplat], axis=1).astype(np.float64)
log(f"피처 {X.shape[1]}개")

seasons = df["season"].to_numpy(np.float64)
tr_m = seasons <= VALID_SEASON - 1
va_m = seasons == VALID_SEASON
yv = y[va_m].astype(np.float64)
w_tr = recency_weight(seasons[tr_m], half_life=2.0)
log(f"train={tr_m.sum():,}  valid={va_m.sum():,}")

results = {}

log("[1] HGB classifier (log_loss)...")
m = HistGradientBoostingClassifier(**HGB_CLS_PARAMS).fit(X.loc[tr_m], y[tr_m], sample_weight=w_tr)
p = m.predict_proba(X.loc[va_m])[:, 1]
results["hgb_logloss"] = score(yv, p)
np.save("phase79_pred_hgb_logloss.npy", p)
log(f"  score={results['hgb_logloss']:.2f}")

log("[2] HGB regressor (squared_error)...")
m = HistGradientBoostingRegressor(**HGB_REG_PARAMS).fit(X.loc[tr_m], y[tr_m].astype(np.float64), sample_weight=w_tr)
p = np.clip(m.predict(X.loc[va_m]), 0, 1)
results["hgb_rmse"] = score(yv, p)
np.save("phase79_pred_hgb_rmse.npy", p)
log(f"  score={results['hgb_rmse']:.2f}")

tr_i, es_i = time_split_es(int(tr_m.sum()))
Xtr = X.loc[tr_m].reset_index(drop=True)
ytr = y[tr_m]
wtr = w_tr

log("[3] CatBoost Logloss (단일시드 42)...")
cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                        random_seed=SEED, verbose=0, early_stopping_rounds=50,
                        min_data_in_leaf=200, loss_function="Logloss")
cb.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=wtr[tr_i], eval_set=(Xtr.iloc[es_i], ytr[es_i]))
p = cb.predict_proba(X.loc[va_m])[:, 1]
results["cat_logloss"] = score(yv, p)
np.save("phase79_pred_cat_logloss.npy", p)
log(f"  best_iter={cb.best_iteration_}  score={results['cat_logloss']:.2f}")

log("[4] CatBoost RMSE (단일시드 42)...")
cbr = CatBoostRegressor(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                        random_seed=SEED, verbose=0, early_stopping_rounds=50,
                        min_data_in_leaf=200, loss_function="RMSE")
cbr.fit(Xtr.iloc[tr_i], ytr[tr_i].astype(np.float64), sample_weight=wtr[tr_i],
       eval_set=(Xtr.iloc[es_i], ytr[es_i].astype(np.float64)))
p = np.clip(cbr.predict(X.loc[va_m]), 0, 1)
results["cat_rmse"] = score(yv, p)
np.save("phase79_pred_cat_rmse.npy", p)
log(f"  best_iter={cbr.best_iteration_}  score={results['cat_rmse']:.2f}")

log("[5] 블렌드 비교 (0.5/0.5)...")
p_hgb_ll = np.load("phase79_pred_hgb_logloss.npy")
p_hgb_rm = np.load("phase79_pred_hgb_rmse.npy")
p_cat_ll = np.load("phase79_pred_cat_logloss.npy")
p_cat_rm = np.load("phase79_pred_cat_rmse.npy")
results["blend_logloss"] = score(yv, 0.5 * p_hgb_ll + 0.5 * p_cat_ll)
results["blend_rmse"] = score(yv, 0.5 * p_hgb_rm + 0.5 * p_cat_rm)
results["blend_mixed_best"] = score(yv, 0.5 * p_hgb_rm + 0.5 * p_cat_ll)

print()
print("=" * 46)
for k, v in results.items():
    print(f"  {k:<20}{v:10.2f}")
print("=" * 46)
print(f"objective 자체 효과 (blend_rmse - blend_logloss): {results['blend_rmse']-results['blend_logloss']:+.2f}")

pd.Series(results).to_csv("phase79_objective.csv")
log(f"총 {time.time()-t0:.0f}s")
