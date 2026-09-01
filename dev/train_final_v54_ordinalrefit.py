"""v54 = v50 + ordinal(3-stage HGB) refit-closure.
v41 이후 3-stage 모두 early_stopping=True, validation_fraction=0.08로 각 stage
데이터의 마지막(내부 자동분할) 8%가 실제 트리분할에 안 쓰였음. ES로 n_iter 확정 후
early_stopping=False로 같은 stage 데이터 전체에 고정반복 재학습.
시간 제약으로 fold A/C 사전 스크리닝 생략 (cats-refit 실측이 fold신호와 반대로
나온 전례가 있어 이번은 로컬검증 없이 직접 실측으로 판단).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
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
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
OUT_DIR = "../submit/model"
t0 = time.time()
_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")


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


log("v50 아티팩트 로드...")
v50 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v50.pkl"))
log(f"  ordinal_weight={v50.get('ordinal_weight')}")

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
X = X[v50["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개")

w_rec = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)

log("투구단위 라벨 복원 (reverse/middle)...")
pid = df["pitcher_id"].to_numpy()
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(df), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(len(df))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(df))
    lab[order] = d
    return lab


lab_reverse = diff_label("asof_pitcher_reverse_rate")
lab_middle = diff_label("asof_pitcher_middle_rate")
valid = ~(np.isnan(lab_reverse) | np.isnan(lab_middle))
log(f"  라벨 유효행 {valid.sum()}/{len(df)} ({valid.mean()*100:.2f}%)")

HGB_ES = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
             early_stopping=True, validation_fraction=0.08, n_iter_no_change=20, random_state=42)

log("stage1: P(not reverse) — ES로 iter 확정 -> refit...")
ts = time.time()
y1 = (1 - lab_reverse[valid])
m1es = HistGradientBoostingClassifier(**HGB_ES)
m1es.fit(X.loc[valid], y1, sample_weight=w_rec[valid])
it1 = m1es.n_iter_
m1 = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=it1, learning_rate=0.03,
                                    l2_regularization=5.0, early_stopping=False, random_state=42)
m1.fit(X.loc[valid], y1, sample_weight=w_rec[valid])
strip_rng(m1)
log(f"  stage1 ES_iter={it1} refit완료 ({time.time()-ts:.0f}s)")

not_rev = valid & (lab_reverse == 0)
log(f"stage2: P(not middle | not reverse) ({not_rev.sum()}행)...")
ts = time.time()
y2 = (1 - lab_middle[not_rev])
m2es = HistGradientBoostingClassifier(**HGB_ES)
m2es.fit(X.loc[not_rev], y2, sample_weight=w_rec[not_rev])
it2 = m2es.n_iter_
m2 = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=it2, learning_rate=0.03,
                                    l2_regularization=5.0, early_stopping=False, random_state=42)
m2.fit(X.loc[not_rev], y2, sample_weight=w_rec[not_rev])
strip_rng(m2)
log(f"  stage2 ES_iter={it2} refit완료 ({time.time()-ts:.0f}s)")

not_rev_mid = not_rev & (lab_middle == 0)
log(f"stage3: P(success | not reverse & not middle) ({not_rev_mid.sum()}행)...")
ts = time.time()
y3 = y[not_rev_mid]
m3es = HistGradientBoostingClassifier(**HGB_ES)
m3es.fit(X.loc[not_rev_mid], y3, sample_weight=w_rec[not_rev_mid])
it3 = m3es.n_iter_
m3 = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=it3, learning_rate=0.03,
                                    l2_regularization=5.0, early_stopping=False, random_state=42)
m3.fit(X.loc[not_rev_mid], y3, sample_weight=w_rec[not_rev_mid])
strip_rng(m3)
log(f"  stage3 ES_iter={it3} refit완료 ({time.time()-ts:.0f}s)")

common = dict(v50)
common["ordinal_stage1"] = m1
common["ordinal_stage2"] = m2
common["ordinal_stage3"] = m3

out = os.path.join(OUT_DIR, "model_artifacts_v54.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
