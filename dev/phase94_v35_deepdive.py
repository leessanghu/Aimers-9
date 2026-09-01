"""phase94 — v35 모델 심층분석: 어떤 피처를 얼마나 쓰고, 어디서 split하고,
어느 구간이 확률을 얼마나 밀어올리는지.

3단계:
    1) SHAP magnitude (CatBoost 3변종 평균) -> 전체 피처 사용량 랭킹
    2) 실제 split 위치 (CatBoost JSON dump 파싱) -> 상위 피처가 '어느 값'에서 갈라지는지
    3) SHAP dependence (값 구간별 평균 SHAP) -> 그 split이 확률을 밀어올리는 방향/크기
"""
import json
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

TM_CACHE = "phase64_trackman_profile.parquet"
SAMPLE_N = 300_000
SEED = 42
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("v35 아티팩트 로드...")
art = joblib.load("../submit/model/model_artifacts_v35.pkl")
cats = art["cats"]
feature_order = art["feature_order"]
log(f"  피처 {len(feature_order)}개, CatBoost {len(cats)}변종")

log("피처 재구성 (train_final_v35와 동일)...")
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

rng = np.random.RandomState(SEED)
recent = np.where(df["season"].to_numpy() >= 2023)[0]
samp = rng.choice(recent, min(SAMPLE_N, len(recent)), replace=False)
samp.sort()
log(f"SHAP 표본 {len(samp):,}행 (2023~2024)")

# ---------------- [1] SHAP magnitude ----------------
log("[1] SHAP 계산 (CatBoost 3변종 평균)...")
mats, signed_mats = [], []
for i, cb in enumerate(cats):
    sv = cb.get_feature_importance(Pool(X.iloc[samp], y[samp]), type="ShapValues")
    mats.append(np.abs(sv[:, :-1]).mean(axis=0))
    signed_mats.append(sv[:, :-1])
    log(f"  seed {i+1}/{len(cats)} 완료")
mag = np.mean(mats, axis=0)
mag_s = pd.Series(mag, index=feature_order).sort_values(ascending=False)
shap_signed = np.mean(signed_mats, axis=0)  # (n_samp, n_feat) 평균 SHAP값

print()
print("=" * 78)
print("SHAP magnitude 상위 25 (모델이 실제로 얼마나 쓰는가)")
print("=" * 78)
cum = mag_s.cumsum() / mag_s.sum()
for i, (k, v) in enumerate(mag_s.head(25).items(), 1):
    print(f"  {i:2d}. {k:<32}{v:.5f}   누적기여 {cum[k]*100:5.1f}%")
print(f"\n상위 20개 = 전체 magnitude의 {cum.iloc[19]*100:.1f}%")
print(f"상위 40개 = 전체 magnitude의 {cum.iloc[39]*100:.1f}%")

# ---------------- [2] 실제 split 위치 (CatBoost JSON dump) ----------------
log("[2] CatBoost 트리 구조 파싱 (split 피처/threshold)...")
cb0 = cats[0]
cb0.save_model("phase94_tmp.json", format="json")
with open("phase94_tmp.json", encoding="utf-8") as f:
    dump = json.load(f)
os.remove("phase94_tmp.json")

fidx_to_name = {ff["flat_feature_index"]: ff["feature_id"] for ff in dump["features_info"]["float_features"]}
split_records = []


def walk_tree(tree):
    splits = tree.get("splits", [])
    for sp in splits:
        fidx = sp.get("float_feature_index")
        border = sp.get("border")
        if fidx is not None and border is not None:
            split_records.append((fidx, border))


for tree in dump["oblivious_trees"]:
    walk_tree(tree)

sdf = pd.DataFrame(split_records, columns=["fidx", "border"])
sdf["feature"] = sdf["fidx"].map(lambda i: fidx_to_name.get(i, f"f{i}"))
split_counts = sdf.groupby("feature").size().sort_values(ascending=False)

print()
print("=" * 78)
print("실제 split 횟수 상위 20 (CatBoost seed1, 트리 299개 전체 누적)")
print("=" * 78)
for i, (k, v) in enumerate(split_counts.head(20).items(), 1):
    borders = sdf[sdf.feature == k]["border"].to_numpy()
    q = np.percentile(borders, [10, 50, 90]) if len(borders) else [np.nan] * 3
    print(f"  {i:2d}. {k:<32}{v:4d}회   threshold 10/50/90% = {q[0]:.4f} / {q[1]:.4f} / {q[2]:.4f}")

# ---------------- [3] SHAP dependence (구간별 평균 SHAP) ----------------
log("[3] 상위 12개 피처의 SHAP dependence (값 구간별 확률 기여)...")
top12 = mag_s.head(12).index.tolist()
Xs = X.iloc[samp].reset_index(drop=True)
print()
print("=" * 78)
print("SHAP dependence: 값이 낮은 구간 -> 높은 구간으로 갈수록 SHAP(로짓기여)이 어떻게 변하나")
print("=" * 78)
dep_rows = []
for feat in top12:
    fi = feature_order.index(feat)
    v = Xs[feat].to_numpy(np.float64)
    sh = shap_signed[:, fi]
    try:
        bins = pd.qcut(v, 10, duplicates="drop")
    except ValueError:
        bins = pd.cut(v, min(10, len(np.unique(v))))
    tmp = pd.DataFrame({"bin": bins, "v": v, "shap": sh})
    g_ = tmp.groupby("bin", observed=True).agg(v_mean=("v", "mean"), shap_mean=("shap", "mean"),
                                               n=("shap", "size"))
    direction = "단조증가" if g_["shap_mean"].is_monotonic_increasing else (
        "단조감소" if g_["shap_mean"].is_monotonic_decreasing else "비단조(임계/U자 가능)")
    spread = g_["shap_mean"].max() - g_["shap_mean"].min()
    print(f"\n  [{feat}]  형태={direction}  SHAP범위폭={spread:.4f}")
    for _, r in g_.iterrows():
        bar = "#" * int(30 * (r.shap_mean - g_.shap_mean.min()) / max(spread, 1e-9))
        print(f"      값={r.v_mean:9.4f}  SHAP={r.shap_mean:+.4f}  n={int(r.n):6d}  {bar}")
    dep_rows.append(dict(feature=feat, direction=direction, shap_spread=spread))

pd.DataFrame(dep_rows).to_csv("phase94_dependence_summary.csv", index=False)
mag_s.to_csv("phase94_shap_magnitude.csv", header=["magnitude"])
split_counts.to_csv("phase94_split_counts.csv", header=["count"])
log(f"저장 완료. 총 {time.time()-t0:.0f}s")
