"""모델 조합 3종 비교 (v26 132피처, train<=2023->valid=2024, 진짜 미지시즌).

조합 1: HGB + CatBoost + MLP           <- phase67 캐시 재사용, 3-way 가중탐색
조합 2: HGB + CatBoost 3seed평균        <- v18 아이디어 재검증 (그때는 hidden_denom과
                                          섞여서 confound됐음. 이론상 분산만 줄어 손해 불가능)
조합 3: HGB + CatBoost + RandomForest   <- boosting 2개 대신 bagging(RF)을 섞어 진짜
                                          다른 학습동역학의 모델을 추가 (phase68에서
                                          capacity를 늘린 boosting은 전부 손해였으므로,
                                          '더 복잡한 boosting'이 아니라 '다른 종류의 모델'을 시도)
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from count_split import build_count_table, transform_count, K_COUNT
from crosses import add_crosses
from career_volatility import build_volatility_table, transform_volatility, K_VOL
from formfeat import build_role_table, transform_form, transform_role
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import build_trackman_profile, transform_trackman

SEED = 42
TRAIN_MAX, VALID_SEASON = 2023, 2024
HALF_LIFE = 2.0
TM_CACHE = "phase64_trackman_profile.parquet"
CACHE = "phase67_cache"
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def bss(y, p):
    r = y.mean()
    return max(0.0, 1e5 * (1 - np.mean((p - y) ** 2) / (r * (1 - r))))


log("피처 구성 (v26과 동일 132개)...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())

se = build_season_end_table(df)
dins = transform_inseason(df, se, g, sr)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
dcnt = transform_count(df, build_count_table(df), pp, sr, k=K_COUNT)
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
dly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0)
dvol = transform_volatility(df, build_volatility_table(se), sr, k=K_VOL)
role_tbl = build_role_table(df)
drole = transform_role(df, role_tbl, sr)
base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
dform = transform_form(df, drole, dins["inseason_success_smooth"].to_numpy(np.float64), base_middle)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
dtm = transform_trackman(df, prof, sr)
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]


def stack(i, base_frame):
    X = pd.concat([base_frame.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    parts = [X, add_crosses(X), dly.loc[i].reset_index(drop=True), dcnt.loc[i].reset_index(drop=True),
             dvol.loc[i].reset_index(drop=True), drole.loc[i].reset_index(drop=True),
             dform.loc[i].reset_index(drop=True), dtm.loc[i].reset_index(drop=True)]
    return pd.concat(parts, axis=1)


tr_i = df[df.season <= TRAIN_MAX].index
va_i = df[df.season == VALID_SEASON].index
fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED,
                  include_team_te=True, team_te_mode="expanding")
y_tr, y_va = fold["y_train"], fold["y_valid"]
X_tr = stack(tr_i, fold["X_train"])
X_va = stack(va_i, fold["X_valid"])
log(f"피처 {X_tr.shape[1]}개  train={len(X_tr):,}  valid={len(X_va):,}")


def recency_weight(seasons, half_life=HALF_LIFE):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


w_tr = recency_weight(df.loc[tr_i, "season"].to_numpy(np.float64))
w_tr = w_tr / w_tr.mean()
ti, ei = time_split_es(len(X_tr))

# ======================================================================
# HGB (한 번만 학습, 세 조합 공통)
# ======================================================================
log("HGB 학습 (공통)...")
h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                   l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=20, random_state=SEED)
h.fit(X_tr, y_tr, sample_weight=w_tr)
p_hgb = h.predict_proba(X_va)[:, 1]
log(f"  HGB 단독 score={bss(y_va, p_hgb):.1f}")

# ======================================================================
# CatBoost 3seed (조합2용, 조합1/3의 seed=42도 재사용)
# ======================================================================
cat_preds = {}
for s in (42, 7, 2024):
    log(f"CatBoost seed={s} 학습...")
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            random_seed=s, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(X_tr.iloc[ti], y_tr[ti], sample_weight=w_tr[ti], eval_set=(X_tr.iloc[ei], y_tr[ei]))
    p = cb.predict_proba(X_va)[:, 1]
    cat_preds[s] = p
    log(f"  seed={s} 단독 score={bss(y_va, p):.1f}  best_iter={cb.best_iteration_}")

p_cat1 = cat_preds[42]
p_cat3 = np.mean([cat_preds[s] for s in (42, 7, 2024)], axis=0)

# ======================================================================
# RandomForest (조합3용)
# ======================================================================
log("RandomForest 학습...")
Xtr_f = X_tr.fillna(0.0).to_numpy(np.float64)
Xva_f = X_va.fillna(0.0).to_numpy(np.float64)
rf = RandomForestClassifier(n_estimators=500, max_depth=12, min_samples_leaf=100,
                            max_features="sqrt", n_jobs=-1, random_state=SEED)
rf.fit(Xtr_f, y_tr, sample_weight=w_tr)
p_rf = rf.predict_proba(Xva_f)[:, 1]
log(f"  RF 단독 score={bss(y_va, p_rf):.1f}")

# ======================================================================
# MLP (phase67 캐시가 있으면 재사용)
# ======================================================================
mlp_path = f"{CACHE}/mlp_v26_valid_pred.npy"
p_mlp = np.load(mlp_path) if os.path.exists(mlp_path) else None
if p_mlp is not None:
    log(f"MLP 캐시 로드 score={bss(y_va, p_mlp):.1f}")
else:
    log("MLP 캐시 없음 -> 조합1은 스킵")

# ======================================================================
# 조합 비교
# ======================================================================
log("\n" + "=" * 78)
log("모델 조합 비교 (2024 폴드, 진짜 미지시즌)")
log("=" * 78)

base_blend = 0.5 * p_hgb + 0.5 * p_cat1
log(f"  기준(HGB 0.5 + CatBoost1seed 0.5)         {bss(y_va, base_blend):8.1f}")

if p_mlp is not None:
    log("\n[조합1] HGB + CatBoost + MLP 가중탐색")
    best1 = (0, None)
    for w_c in (0.3, 0.4, 0.5, 0.6):
        for w_m in (0.05, 0.1, 0.15, 0.2, 0.25):
            w_h = 1 - w_c - w_m
            if w_h < 0:
                continue
            p = w_h * p_hgb + w_c * p_cat1 + w_m * p_mlp
            sc = bss(y_va, p)
            if sc > best1[0]:
                best1 = (sc, (w_h, w_c, w_m))
    log(f"  최적: HGB={best1[1][0]:.2f} CatBoost={best1[1][1]:.2f} MLP={best1[1][2]:.2f}  "
        f"score={best1[0]:.1f}  (기준 대비 {best1[0]-bss(y_va,base_blend):+.1f})")

log("\n[조합2] HGB + CatBoost 3seed평균")
for w_c in (0.4, 0.5, 0.6):
    p = (1 - w_c) * p_hgb + w_c * p_cat3
    log(f"  HGB={1-w_c:.1f} Cat3seed={w_c:.1f}   score={bss(y_va, p):8.1f}  "
        f"(기준 대비 {bss(y_va,p)-bss(y_va,base_blend):+.1f})")

log("\n[조합3] HGB + CatBoost + RandomForest 가중탐색")
best3 = (0, None)
for w_c in (0.3, 0.4, 0.5, 0.6):
    for w_r in (0.05, 0.1, 0.15, 0.2, 0.25):
        w_h = 1 - w_c - w_r
        if w_h < 0:
            continue
        p = w_h * p_hgb + w_c * p_cat1 + w_r * p_rf
        sc = bss(y_va, p)
        if sc > best3[0]:
            best3 = (sc, (w_h, w_c, w_r))
log(f"  최적: HGB={best3[1][0]:.2f} CatBoost={best3[1][1]:.2f} RF={best3[1][2]:.2f}  "
    f"score={best3[0]:.1f}  (기준 대비 {best3[0]-bss(y_va,base_blend):+.1f})")

log("\n" + "=" * 78)
log("요약")
log("=" * 78)
log(f"  기준 (HGB+CatBoost 50:50)     {bss(y_va, base_blend):8.1f}")
if p_mlp is not None:
    log(f"  조합1 (+MLP)                  {best1[0]:8.1f}  ({best1[0]-bss(y_va,base_blend):+.1f})")
log(f"  조합2 (+Cat 3seed)             {bss(y_va, 0.5*p_hgb+0.5*p_cat3):8.1f}  "
    f"({bss(y_va,0.5*p_hgb+0.5*p_cat3)-bss(y_va,base_blend):+.1f})")
log(f"  조합3 (+RandomForest)          {best3[0]:8.1f}  ({best3[0]-bss(y_va,base_blend):+.1f})")

np.savez(f"{CACHE}/phase69_preds.npz", hgb=p_hgb, cat1=p_cat1, cat3=p_cat3, rf=p_rf,
        mlp=p_mlp if p_mlp is not None else np.zeros_like(p_hgb), y=y_va)
log(f"\n총 {time.time()-t0:.0f}s")
