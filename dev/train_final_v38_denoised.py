"""v38 = v35(재사용) + Cross-fitted Denoised Target (idea5, 3폴드 검증 통과).

검증(idea5_denoised_target.py): HGB d6 하나로 A/C/B 3폴드, w 세밀스캔 결과
    w=0.14: A+6.29 C+7.56 B+276.86  (최소이득 +6.29, 전부 플러스)
메커니즘: fold B(regime단절 스트레스폴드)에서 base λ=0.286(극심한 과대분산,
dispersion손실 930점) vs denoised λ=0.772(37.6점). rho(진짜y)도 0.0387->0.0656로
거의 2배 -- 캘리브레이션 우연이 아니라 y_soft의 낮은 분산 자체가 정규화 역할.

프로덕션: 전체 2019~2024 데이터에서 LOO(자기 행 제외) empirical Bayes로 y_soft 생성 후
HistGradientBoostingRegressor 학습, v35 블렌드에 w=0.14로 추가.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

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
K_PS = 15.0
K_CELL = 30.0
DENOISE_WEIGHT = 0.14


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
log(f"  hgbs={len(v35['hgbs'])} cats={len(v35['cats'])} core={len(v35['core_fail_models'])}")

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
seasons = df["season"].to_numpy(np.float64)
pid = df["pitcher_id"].to_numpy()
same_hand = X["same_hand"].to_numpy(np.float64)
balls = df["balls_before"].to_numpy(np.float64)
strikes = df["strikes_before"].to_numpy(np.float64)
pressure_bucket = np.sign(balls - strikes).astype(np.int64)

log("y_soft 생성 (전체데이터 LOO empirical Bayes)...")
sub = pd.DataFrame({"pid": pid, "season": seasons, "y": y, "pb": pressure_bucket, "sh": same_hand})
ps_grp = sub.groupby(["pid", "season"])["y"].agg(ps_sum="sum", ps_n="count")
sub = sub.join(ps_grp, on=["pid", "season"])
ps_sum_loo = sub["ps_sum"] - sub["y"]
ps_n_loo = sub["ps_n"] - 1
p_ps_loo = (ps_sum_loo + K_PS * g) / (ps_n_loo + K_PS)

cell_grp = sub.groupby(["pid", "season", "pb", "sh"])["y"].agg(c_sum="sum", c_n="count")
sub = sub.join(cell_grp, on=["pid", "season", "pb", "sh"])
c_sum_loo = sub["c_sum"] - sub["y"]
c_n_loo = sub["c_n"] - 1
y_soft = ((c_sum_loo + K_CELL * p_ps_loo) / (c_n_loo + K_CELL)).to_numpy(np.float64)
log(f"  y_soft mean={y_soft.mean():.4f} std={y_soft.std():.4f} (원 y std={y.std():.4f})")

log("Denoised HGB 회귀모델 학습 (전체데이터)...")
HGB_REG = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
              early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42,
              loss="squared_error")
ts = time.time()
denoise_model = HistGradientBoostingRegressor(**HGB_REG).fit(X, y_soft, sample_weight=w_rec)
log(f"  학습완료 iters={denoise_model.n_iter_} ({time.time()-ts:.0f}s)")
strip_rng(denoise_model)

common = dict(v35)
common["denoise_model"] = denoise_model
common["denoise_weight"] = DENOISE_WEIGHT
out = os.path.join(OUT_DIR, "model_artifacts_v38.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
