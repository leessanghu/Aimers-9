"""v41 = v35(재사용) + 3단 순서형 Hurdle 캐스케이드 (아이디어A, 시간압박으로 로컬검증 생략 즉시실측).

P(success) = P(not reverse) * P(not middle | not reverse) * P(success | not reverse & not middle)
전부 "전체 모집단의 결과값(타겟) 기준 부분집합"이라 표본은 여전히 크다(엔티티별로
안 쪼갠다는 원칙 유지) -- Hurdle(core_fail x success|no_core)의 3단 확장판.
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
from pitchlabels import recover_pitch_labels
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
OUT_DIR = "../submit/model"
t0 = time.time()
_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")
ORDINAL_WEIGHT = 0.15


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


log("v35 아티팩트 로드...")
v35 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v35.pkl"))
log(f"  hgbs={len(v35['hgbs'])} cats={len(v35['cats'])} core={len(v35['core_fail_models'])}")

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
X = X[v35["feature_order"]].astype(np.float64)
log(f"피처 {X.shape[1]}개")

w_rec = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)

log("투구단위 라벨 복원 (reverse/middle)...")
labels = recover_pitch_labels(df)
lab_reverse = labels["lab_reverse"].to_numpy(np.float64)
lab_middle = labels["lab_middle"].to_numpy(np.float64)
valid = ~(np.isnan(lab_reverse) | np.isnan(lab_middle))
log(f"  라벨 유효행 {valid.sum()}/{len(df)} ({valid.mean()*100:.2f}%)")

HGB_CLS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
              early_stopping=True, validation_fraction=0.08, n_iter_no_change=20, random_state=42)

log("stage1: P(not reverse) 학습 (전체 유효행)...")
m1 = HistGradientBoostingClassifier(**HGB_CLS)
ts = time.time()
y1 = (1 - lab_reverse[valid])
m1.fit(X.loc[valid], y1, sample_weight=w_rec[valid])
log(f"  stage1 완료 iters={m1.n_iter_} ({time.time()-ts:.0f}s)")
strip_rng(m1)

not_rev = valid & (lab_reverse == 0)
log(f"stage2: P(not middle | not reverse) 학습 ({not_rev.sum()}행)...")
m2 = HistGradientBoostingClassifier(**HGB_CLS)
ts = time.time()
y2 = (1 - lab_middle[not_rev])
m2.fit(X.loc[not_rev], y2, sample_weight=w_rec[not_rev])
log(f"  stage2 완료 iters={m2.n_iter_} ({time.time()-ts:.0f}s)")
strip_rng(m2)

not_rev_mid = not_rev & (lab_middle == 0)
log(f"stage3: P(success | not reverse & not middle) 학습 ({not_rev_mid.sum()}행)...")
m3 = HistGradientBoostingClassifier(**HGB_CLS)
ts = time.time()
y3 = y[not_rev_mid]
m3.fit(X.loc[not_rev_mid], y3, sample_weight=w_rec[not_rev_mid])
log(f"  stage3 완료 iters={m3.n_iter_} ({time.time()-ts:.0f}s)")
strip_rng(m3)

p1 = m1.predict_proba(X)[:, 1]
p2 = m2.predict_proba(X)[:, 1]
p3 = m3.predict_proba(X)[:, 1]
p_ordinal = p1 * p2 * p3
log(f"in-sample 캐스케이드 확인: mean(p_ordinal)={p_ordinal.mean():.4f} vs 실제 success율={g:.4f}")

common = dict(v35)
common["ordinal_stage1"] = m1
common["ordinal_stage2"] = m2
common["ordinal_stage3"] = m3
common["ordinal_weight"] = ORDINAL_WEIGHT
hurdle_w = common.get("hurdle_weight", 0.0)
mix_w = common.get("mix_weight", 0.0)
denoise_w = common.get("denoise_weight", 0.0)
multi_w = common.get("multi_weight", 0.0)
common["base_weight"] = 1.0 - hurdle_w - mix_w - denoise_w - multi_w - ORDINAL_WEIGHT
log(f"weights: base={common['base_weight']:.3f} hurdle={hurdle_w:.2f} ordinal={ORDINAL_WEIGHT:.2f}")

out = os.path.join(OUT_DIR, "model_artifacts_v41.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
