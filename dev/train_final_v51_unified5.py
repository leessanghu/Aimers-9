"""v51 = v47 + 통합 5-head 공유트리 (idea12 재도전).
당시 fold A +0.78을 "시드폭 19.28이라 노이즈"로 기각했으나, 이후 확립된 aux head
편향규칙(fold A -> 실측 +6.15, n=3에서 +6.56/+5.24/+6.64로 일관)을 적용하면
보정추정 +6.9 -- middle축(+1.08 -> 실측 +7.72)과 같은 프로필.
head0=y / h1=not_reverse / h2=not_middle|not_reverse(마스킹) / h3=투수시즌LOO /
h4=투수x손LOO. 추론시엔 head0만 사용.
Rule.md §4 준수: 라벨은 train 누적통계 차분으로만 복원, test 행간 참조 없음.
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
UNIFIED5_WEIGHT = 0.10
K_PS = 15.0
WINDOW = 50


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


log("v47 아티팩트 로드...")
v47 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v47.pkl"))
log(f"  hgbs={len(v47['hgbs'])} cats={len(v47['cats'])}")

log("데이터 로드 + 피처 재구성...")
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
X = X[v47["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개")

w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)

log("투구단위 라벨 복원 (reverse/middle 차분)...")
pid = df["pitcher_id"].to_numpy()
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(df), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def recover(rate_col):
    c = np.round(df[rate_col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(len(df))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(df))
    lab[order] = d
    return lab


lab_reverse = recover("asof_pitcher_reverse_rate")
lab_middle = recover("asof_pitcher_middle_rate")
valid_lab = ~(np.isnan(lab_reverse) | np.isnan(lab_middle))
log(f"  라벨 유효행 {valid_lab.sum():,}/{len(df):,} ({valid_lab.mean()*100:.2f}%)")

log(f"투수-시즌 / 투수x손 LOO율 구성 (K_PS={K_PS})...")
same_hand = X["same_hand"].to_numpy(np.float64)
seasons_arr = df["season"].to_numpy(np.float64)
sub = pd.DataFrame({"pid": pid, "season": seasons_arr, "sh": same_hand, "y": y})
ps = sub.groupby(["pid", "season"])["y"].agg(s="sum", n="count")
sub = sub.join(ps, on=["pid", "season"])
h3 = ((sub["s"] - sub["y"]) + K_PS * g) / ((sub["n"] - 1) + K_PS)
psh = sub.groupby(["pid", "season", "sh"])["y"].agg(s2="sum", n2="count")
sub = sub.join(psh, on=["pid", "season", "sh"])
h4 = ((sub["s2"] - sub["y"]) + K_PS * h3) / ((sub["n2"] - 1) + K_PS)
h3 = h3.to_numpy(np.float64); h4 = h4.to_numpy(np.float64)

# 5-head: y / not_reverse / not_middle|not_reverse(마스킹) / 투수시즌LOO / 투수x손LOO
h1 = np.where(valid_lab, 1.0 - lab_reverse, np.nan)
h2 = np.where(valid_lab & (lab_reverse == 0), 1.0 - lab_middle, np.nan)
log(f"  h1(not_rev) 유효 {np.isfinite(h1).mean()*100:.1f}%  h2(not_mid|not_rev) 유효 {np.isfinite(h2).mean()*100:.1f}%")

Ymat = np.column_stack([y.astype(np.float64), h1, h2, h3, h4])

log("통합 5-head 공유트리 CatBoost 학습 (전체데이터)...")
tr_i, es_i = np.arange(int(len(X) * 0.92)), np.arange(int(len(X) * 0.92), len(X))
CAT_PARAMS = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  random_seed=42, loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)
ts = time.time()
unified5_model = CatBoostRegressor(**CAT_PARAMS)
unified5_model.fit(X.iloc[tr_i], Ymat[tr_i], sample_weight=w[tr_i],
                   eval_set=(X.iloc[es_i], Ymat[es_i]))
log(f"  학습완료 best_iter={unified5_model.best_iteration_} ({time.time()-ts:.0f}s)")
strip_rng(unified5_model)

common = dict(v47)
common["unified5_model"] = unified5_model
common["unified5_weight"] = UNIFIED5_WEIGHT
existing = sum(common.get(k, 0.0) for k in
              ["hurdle_weight", "mix_weight", "denoise_weight", "multi_weight", "multires_weight", "ordinal_weight"])
common["base_weight"] = 1.0 - existing - UNIFIED5_WEIGHT
log(f"weights: base={common['base_weight']:.3f} unified5={UNIFIED5_WEIGHT:.2f} (기존합={existing:.2f})")

out = os.path.join(OUT_DIR, "model_artifacts_v51.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
