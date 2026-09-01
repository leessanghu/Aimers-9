"""v36 = v35(재사용) + 판정축 혼합분해 (phase90 3폴드 검증: 세 폴드 모두 플러스).

가중치 재탐색 결과 (a=hurdle, b=mix, 세 폴드 A/C/B 전부 이득>=0인 조합 중 최고):
    (a=0.55, b=0.15) -> A+2.50 C+8.93 B+2.84 (평균 +4.76)
    base_weight=0.30, hurdle_weight=0.55, mix_weight=0.15

v35의 hgbs/cats/core_fail_models/succ_nc_models을 그대로 재사용하고,
판정축 모델(call 3-class + success|call 3개)만 전체데이터로 신규 학습한다.
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


log("v35 아티팩트 로드 (HGB/CatBoost/Hurdle 재사용)...")
v35 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v35.pkl"))
log(f"  hgbs={len(v35['hgbs'])} cats={len(v35['cats'])} "
    f"core={len(v35['core_fail_models'])} snc={len(v35['succ_nc_models'])}")

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

w = recency_weight(df["season"].to_numpy(np.float64), half_life=2.0)

log("판정축 라벨 복원 (call: ball/strike/inplay)...")
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)


def cnt(col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)


B_, K_ = [cnt(c) for c in ["asof_pitcher_ball_rate", "asof_pitcher_strike_rate"]]
ordr = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = df["pitcher_id"].to_numpy()[ordr]
n_o = n_[ordr]
step = np.zeros(len(df), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
d_b = np.zeros(len(df)); d_k = np.zeros(len(df))
d_b[ordr[:-1]] = np.diff(B_[ordr]); d_k[ordr[:-1]] = np.diff(K_[ordr])
call = np.full(len(df), np.nan)
call[step & (d_b > 0)] = 0
call[step & (d_k > 0)] = 1
call[step & (d_b == 0) & (d_k == 0)] = 2
log(f"  복원 {step.sum():,}행  ball={np.nanmean(call==0):.3f} strike={np.nanmean(call==1):.3f} "
    f"inplay={np.nanmean(call==2):.3f}")

HGB_D6 = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
             early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)

log("call 3-class 모델 학습...")
call3_model = HistGradientBoostingClassifier(**HGB_D6).fit(X.loc[step], call[step], sample_weight=w[step])
log(f"  iters={call3_model.n_iter_}")

succ_given_call_models = []
for c, cname in [(0, "ball"), (1, "strike"), (2, "inplay")]:
    m_c = step & (call == c)
    sm = HistGradientBoostingClassifier(**HGB_D6).fit(X.loc[m_c], y[m_c], sample_weight=w[m_c])
    succ_given_call_models.append(sm)
    log(f"  succ|{cname} iters={sm.n_iter_} (학습행 {m_c.sum():,})")

strip_rng(call3_model)
for m in succ_given_call_models:
    strip_rng(m)

common = dict(v35)
common["call3_model"] = call3_model
common["succ_given_call_models"] = succ_given_call_models
common["base_weight"] = 0.30
common["hurdle_weight"] = 0.55
common["mix_weight"] = 0.15
out = os.path.join(OUT_DIR, "model_artifacts_v36.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
