"""v39 = v29베이스(HGB3+CatBoost3, v35서 재사용) + Masked Multi-Task 공유트리 (idea6).

검증(idea6_multitask_shared_tree.py): v29local(893.68 상당) 기준 A/C/B 3폴드
    w=0.1: A+8.69 C+11.29 B+96.58   최소 +8.69
    w=0.2: A+15.14 C+8.60 B+189.01  최소 +8.60  <- 채택 (최소이득 최고)
    w=0.3: fold C -8.07로 기각

주의: 검증은 v29local(순수 HGB3+CatBoost3) 기준이었지 v35(Hurdle 포함) 기준이 아니다.
멀티태스크 자체가 head1(core_fail)/head2(success|no_core)로 hurdle 기능을 내장하므로,
v35의 별도 Hurdle 모델은 빼고 순수 v29 베이스 위에 이 멀티태스크만 얹는다.

    p_ensemble = 0.5*HGB3평균 + 0.5*CatBoost3평균  (v29와 동일)
    p_multi    = 0.5*clip(head0) + 0.5*(1-clip(head1))*clip(head2)
    최종        = 0.8*p_ensemble + 0.2*p_multi
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

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
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
OUT_DIR = "../submit/model"
t0 = time.time()
_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")
MULTI_WEIGHT = 0.2


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG_TYPES:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strip_rng(v, seen, depth + 1)


def recency_weight(seasons, half_life=2.0):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


log("v35 아티팩트 로드 (HGB/CatBoost 베이스만 재사용, Hurdle은 안 씀)...")
v35 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v35.pkl"))
log(f"  hgbs={len(v35['hgbs'])} cats={len(v35['cats'])}")

log("데이터 로드 + 피처 재구성 (동일 162개)...")
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
X = X[v35["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개 (v35 순서 일치)")

w_rec = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = df["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(df), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)
R_ = np.round(df["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_)
M_ = np.round(df["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_)
d_r = np.zeros(len(df)); d_m = np.zeros(len(df))
d_r[ordr[:-1]] = np.diff(R_[ordr]); d_m[ordr[:-1]] = np.diff(M_[ordr])
core_fail = np.where(step, ((d_r > 0) | (d_m > 0)).astype(np.float64), np.nan)
assert (y[step & (core_fail == 1)] == 0).all()
log(f"  core_fail 복원 {step.sum():,}행")

Y0 = y.astype(np.float64)
Y1 = core_fail.copy()
Y2 = np.where(core_fail == 0, y, np.nan)
Ymat = np.column_stack([Y0, Y1, Y2])

tr_i = np.where(step)[0]
es_cut = int(len(tr_i) * 0.92)
tr_sub, es_sub = tr_i[:es_cut], tr_i[es_cut:]

log("Masked Multi-Task CatBoost 학습 (전체데이터, MultiRMSEWithMissingValues)...")
CAT_PARAMS = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  random_seed=42, loss_function="MultiRMSEWithMissingValues",
                  early_stopping_rounds=50)
ts = time.time()
multi_model = CatBoostRegressor(**CAT_PARAMS)
multi_model.fit(X.iloc[tr_sub], Ymat[tr_sub], sample_weight=w_rec[tr_sub],
               eval_set=(X.iloc[es_sub], Ymat[es_sub]))
log(f"  학습완료 best_iter={multi_model.best_iteration_} ({time.time()-ts:.0f}s)")

hgbs = v35["hgbs"]
cats = v35["cats"]
for m in hgbs:
    strip_rng(m)

common = {
    "hgbs": hgbs, "cats": cats, "w_hgb": 0.5, "w_cat": 0.5,
    "multitask_model": multi_model, "multi_weight": MULTI_WEIGHT, "base_weight": 1.0 - MULTI_WEIGHT,
    "stats": v35["stats"], "inseason_stats": v35["inseason_stats"], "platoon_stats": v35["platoon_stats"],
    "inning_stats": v35["inning_stats"], "count_stats": v35["count_stats"],
    "pitchtype_stats": v35["pitchtype_stats"], "lastyear_stats": v35["lastyear_stats"],
    "volatility_stats": v35["volatility_stats"], "role_stats": v35["role_stats"],
    "trackman_stats": v35["trackman_stats"], "batter_stats": v35["batter_stats"],
    "inseason_full_stats": v35["inseason_full_stats"], "batter_split_stats": v35["batter_split_stats"],
    "form_base_middle_global": v35["form_base_middle_global"], "feature_order": v35["feature_order"],
}
out = os.path.join(OUT_DIR, "model_artifacts_v39.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
