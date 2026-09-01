"""v28 학습 결과 검증 — 모델이 신규 피처를 '실제로 쓰고 있는지' SHAP magnitude로 확인.

패키징 전 필수 확인. 스크리너의 증분 잠재력은 '선형 회귀로 잴 때의 상한'이라,
실제 트리가 그 정보를 split으로 활용하지 못하면 무의미하다. 그래서 학습된 CatBoost에서
직접 SHAP을 뽑아 기존 강피처와 같은 스케일로 비교한다.

기준선 (v18/v20/v22 시절 측정치):
    inseason_success_smooth   0.030   <- 이 클래스에 들어가야 '진짜 쓰이는' 피처
    inseason_reverse_smooth   0.022
    platoon_diff              0.004
    (간접 대리지표들)          0.0004~0.0016  <- 여기 머물면 사실상 안 쓰이는 것

판정 기준:
    신규 피처가 0.004(platoon_diff) 이상이면 확실히 활용됨
    0.001 미만이면 트리가 사실상 무시하는 것 -> 증분 잠재력이 실현 안 될 가능성
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from catboost import Pool

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

ARTIFACT = "../submit/model/model_artifacts_v28.pkl"
TM_CACHE = "phase64_trackman_profile.parquet"
SEED = 42
SAMPLE_N = 300_000   # SHAP은 비싸므로 표본 사용 (magnitude 비교엔 충분)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("아티팩트 로드...")
art = joblib.load(ARTIFACT)
cats = art["cats"]
feature_order = art["feature_order"]
log(f"  피처 {len(feature_order)}개, CatBoost {len(cats)}시드")

log("피처 재구성 (train_final_v28.py와 동일)...")
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
               X_isf, X_bmid, X_bplat], axis=1)
X = X[feature_order].astype(np.float64)
log(f"피처 {X.shape[1]}개 재구성 완료")

# 최근 시즌 위주로 표본 (모델이 실제 예측할 분포에 가까움)
rng = np.random.RandomState(SEED)
recent = np.where(df["season"].to_numpy() >= 2023)[0]
samp = rng.choice(recent, min(SAMPLE_N, len(recent)), replace=False)
samp.sort()
log(f"SHAP 표본 {len(samp):,}행 (2023~2024)")

log("SHAP 계산 (CatBoost 3시드 평균)...")
mats = []
for i, cb in enumerate(cats):
    sv = cb.get_feature_importance(Pool(X.iloc[samp], y[samp]), type="ShapValues")
    mats.append(np.abs(sv[:, :-1]).mean(axis=0))
    log(f"  seed {i+1}/{len(cats)} 완료")
mag = np.mean(mats, axis=0)
mag_s = pd.Series(mag, index=feature_order).sort_values(ascending=False)

NEW_V28 = ["inseason_middle_smooth", "inseason_strike_smooth", "inseason_cmd_index",
           "inseason_middle_minus_career", "bat_inseason_middle", "bat_middle_minus_career",
           "bplatoon_diff", "bplatoon_n"]
NEW_V27 = ["bat_inseason_smooth", "bat_inseason_minus_career", "tm_lown_flag"]
REF = ["inseason_success_smooth", "inseason_reverse_smooth", "platoon_diff",
       "count_diff", "x_ability_here", "season", "cat_game_type"]

print()
print("=" * 74)
print("전체 magnitude 상위 20")
print("=" * 74)
for i, (k, v) in enumerate(mag_s.head(20).items(), 1):
    tag = "  <- v28 신규" if k in NEW_V28 else ("  <- v27" if k in NEW_V27 else "")
    print(f"  {i:2d}. {k:<34}{v:.5f}{tag}")

print()
print("=" * 74)
print("v28 신규 피처 검증 (판정: >=0.004 확실히 활용 / <0.001 사실상 무시)")
print("=" * 74)
print(f"{'피처':<34}{'magnitude':>12}{'순위':>7}   판정")
print("-" * 70)
rank = {k: i + 1 for i, k in enumerate(mag_s.index)}
for f in NEW_V28:
    if f not in mag_s.index:
        print(f"{f:<34}{'없음':>12}")
        continue
    v = mag_s[f]
    verdict = "확실히 활용" if v >= 0.004 else ("활용됨" if v >= 0.001 else "거의 무시")
    print(f"{f:<34}{v:12.5f}{rank[f]:7d}   {verdict}")

print()
print("참조 (기존 피처 같은 스케일):")
for f in REF:
    if f in mag_s.index:
        print(f"  {f:<34}{mag_s[f]:12.5f}{rank[f]:7d}")

print()
print("v27 신규 (지난 라운드, 실측 +18.1 기여):")
for f in NEW_V27:
    if f in mag_s.index:
        print(f"  {f:<34}{mag_s[f]:12.5f}{rank[f]:7d}")

mag_s.to_csv("phase75_v28_shap_magnitude.csv", header=["magnitude"])
log(f"\n저장: phase75_v28_shap_magnitude.csv")
log(f"총 {time.time()-t0:.0f}s")
