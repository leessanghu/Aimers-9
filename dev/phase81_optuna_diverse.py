"""phase81 — optuna로 LightGBM/XGBoost 공정 튜닝 + greedy 앙상블 이득 노이즈 검증.

phase80에서 lgb/xgb가 학습 88~90s만에 조기종료돼 최약체로 나왔다(785.9, 777.7).
CatBoost/HGB는 200~500s를 썼다 -> 비교가 불공정했다. optuna로 depth/leaves/lr/reg를
탐색해 공정하게 준다. 목적함수는 개별 점수가 아니라 '현재 greedy 앙상블(893.68,
hgb_d6/d8/sub + cat_d6/d8/rsm)에 추가했을 때 한계 이득' -> phase80의 결론(교차계열이
동일계열보다 낫다)과 일치하는 기준으로 최적화한다.

노이즈 방어: 목적함수 평가는 시드 2개 평균 (phase76b가 단일평가 SD 9~40임을 보임).

phase80의 greedy +12.98도 단일폴드 선택이라 과적합 위험 -> 여기서 시드 반복으로
같이 검증한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

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

optuna.logging.set_verbosity(optuna.logging.WARNING)

TM_CACHE = "phase64_trackman_profile.parquet"
VALID_SEASON = 2024
CACHE_DIR = "phase80_cache"
N_TRIALS = 20
EVAL_SEEDS = [42, 7]

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


log("데이터 로드 + 피처 재구성 (v28 162개, phase80과 동일)...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy()
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())

fb = FeatureBuilder(seed=42, include_raw_rates=False, team_te_mode="expanding").fit(df)
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
r = yv.mean()
BSREF = r * (1 - r)
Xtr = X.loc[tr_m].reset_index(drop=True)
ytr = y[tr_m]
Xva = X.loc[va_m]
tr_i, es_i = time_split_es(int(tr_m.sum()))
log(f"train={tr_m.sum():,}  valid={va_m.sum():,}")


def score(p):
    return 1e5 * (1 - np.mean((p - yv) ** 2) / BSREF)


# phase80 greedy 앙상블 (기준 블렌드) — 선택 횟수 비율을 가중치로 재현 (9:9:8:8:5:1)
weights = {"hgb_d6": 9, "hgb_sub": 9, "cat_d6": 8, "hgb_d8": 8, "cat_d8": 5, "cat_rsm": 1}
tot = sum(weights.values())
p_base = np.zeros(len(yv))
for k, w in weights.items():
    p_base += (w / tot) * np.load(f"{CACHE_DIR}/{k}.npy")
log(f"기준 greedy 블렌드 재현 score={score(p_base):.2f} (phase80: 893.68)")


def fit_lgb(params, seed):
    import lightgbm as lgb
    m = lgb.LGBMClassifier(n_estimators=3000, learning_rate=params["lr"], num_leaves=params["num_leaves"],
                           max_depth=params["max_depth"], reg_lambda=params["reg_lambda"],
                           min_child_samples=params["min_child_samples"],
                           colsample_bytree=params["colsample"], subsample=params["subsample"],
                           subsample_freq=1, random_state=seed, verbose=-1)
    m.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=w_tr[tr_i],
         eval_set=[(Xtr.iloc[es_i], ytr[es_i])],
         callbacks=[lgb.early_stopping(80, verbose=False)])
    return m.predict_proba(Xva)[:, 1]


def fit_xgb(params, seed):
    import xgboost as xgb
    m = xgb.XGBClassifier(n_estimators=3000, learning_rate=params["lr"], max_depth=params["max_depth"],
                          reg_lambda=params["reg_lambda"], min_child_weight=params["min_child_weight"],
                          colsample_bytree=params["colsample"], subsample=params["subsample"],
                          random_state=seed, early_stopping_rounds=80, eval_metric="logloss",
                          tree_method="hist")
    m.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=w_tr[tr_i],
         eval_set=[(Xtr.iloc[es_i], ytr[es_i])], verbose=False)
    return m.predict_proba(Xva)[:, 1]


def make_objective(fit_fn, tag):
    def obj(trial):
        params = dict(
            lr=trial.suggest_float("lr", 0.02, 0.08, log=True),
            max_depth=trial.suggest_int("max_depth", 4, 8),
            reg_lambda=trial.suggest_float("reg_lambda", 1.0, 20.0, log=True),
            colsample=trial.suggest_float("colsample", 0.5, 1.0),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
        )
        if tag == "lgb":
            params["num_leaves"] = trial.suggest_int("num_leaves", 15, 127)
            params["min_child_samples"] = trial.suggest_int("min_child_samples", 50, 500)
        else:
            params["min_child_weight"] = trial.suggest_float("min_child_weight", 1.0, 50.0, log=True)
        gains = []
        for s in EVAL_SEEDS:
            p = fit_fn(params, s)
            gain = score(0.5 * p_base + 0.5 * p) - score(p_base)
            gains.append(gain)
        m = float(np.mean(gains))
        trial.set_user_attr("gains", gains)
        return m
    return obj


for tag, fit_fn in [("lgb", fit_lgb), ("xgb", fit_xgb)]:
    log(f"[optuna] {tag} 튜닝 시작 ({N_TRIALS} trials, 시드 {EVAL_SEEDS})...")
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(make_objective(fit_fn, tag), n_trials=N_TRIALS, show_progress_bar=False)
    log(f"  {tag} 최고 한계이득={study.best_value:+.2f}  params={study.best_params}")
    study.trials_dataframe().to_csv(f"phase81_{tag}_trials.csv", index=False)

    ts = time.time()
    bp = study.best_params
    seed_preds = []
    for s in [42, 7, 2024]:
        p = fit_fn(bp, s)
        seed_preds.append(p)
        np.save(f"{CACHE_DIR}/{tag}_tuned_seed{s}.npy", p)
    p_avg = np.mean(seed_preds, axis=0)
    np.save(f"{CACHE_DIR}/{tag}_tuned.npy", p_avg)
    log(f"  {tag} 최종(3시드평균) 개별점수={score(p_avg):.2f}  "
       f"blend기여={score(0.5*p_base+0.5*p_avg)-score(p_base):+.2f}  ({time.time()-ts:.0f}s)")

log(f"총 {time.time()-t0:.0f}s")
