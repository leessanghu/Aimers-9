"""CatBoost — Optuna 없이 보수적 고정 파라미터로 1회만 학습 (RF/HGB급 정규화 수준 맞춤).

1) 3개 폴드(2022/2023/2024)로 빠른 OOF 체크
2) 전체 2019~2024로 최종 학습 -> submit 패키징용 모델 저장
raw pitcher_id/batter_id categorical + D 피처셋(pruned+신규2, team_te 제거) 사용.
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from features import FeatureBuilder, TARGET_COL
from metrics import evaluate
from phase2_common import FOLDS, build_fold, load_full, time_split_es

SEED = 42
EXTRA_FEATURES = {"count_asof_ball", "diff_prev1_prev5"}
DEAD_LIST_EXCL_SEASON = [
    c for c in pd.read_csv("dead_features_conservative_list.csv")["feature"].tolist()
    if c != "season"
]
CAT_FEATURES_BASE = ["cat_top_bottom", "cat_game_type", "cat_base_state", "count_state", "hand_matchup"]
ID_COLS = ["pitcher_id", "batter_id"]

# 보수적 고정 파라미터 (RF max_depth=10/min_leaf=200, HGB max_depth=6/l2=5.0 수준에 맞춤)
CB_PARAMS = dict(
    iterations=2000, depth=6, learning_rate=0.03, l2_leaf_reg=10.0,
    random_strength=1.0, bagging_temperature=1.0, border_count=128,
    random_seed=SEED, thread_count=-1, verbose=0, allow_writing_files=False,
    early_stopping_rounds=100, has_time=True,
)


def build_cat_matrix(X_raw, id_df):
    X = X_raw.drop(columns=[c for c in DEAD_LIST_EXCL_SEASON if c in X_raw.columns]).copy()
    X["pitcher_id"] = id_df["pitcher_id"].to_numpy()
    X["batter_id"] = id_df["batter_id"].to_numpy()
    cat_features = [c for c in CAT_FEATURES_BASE if c in X.columns] + ID_COLS
    for c in cat_features:
        X[c] = X[c].astype(np.int64).astype(str)
    return X, cat_features


def main():
    t0 = time.time()
    df = load_full()

    print("===== 3폴드 OOF 체크 (보수적 고정 파라미터) =====", flush=True)
    for train_max, valid_season in FOLDS:
        tm = time.time()
        fold = build_fold(df, train_max, valid_season, extra_features=EXTRA_FEATURES, seed=SEED,
                          include_team_te=False)
        X_train, cat_features = build_cat_matrix(fold["X_train"], fold["train_fold"])
        X_valid, _ = build_cat_matrix(fold["X_valid"], fold["valid_fold"])
        y_train, y_valid = fold["y_train"], fold["y_valid"]
        tr_idx, es_idx = time_split_es(len(X_train))

        pool_tr = Pool(X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64), cat_features=cat_features)
        pool_es = Pool(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64), cat_features=cat_features)
        m = CatBoostRegressor(loss_function="RMSE", **CB_PARAMS)
        m.fit(pool_tr, eval_set=pool_es)
        pred = np.clip(m.predict(Pool(X_valid, cat_features=cat_features)), 0.0, 1.0)
        bss = evaluate(y_valid, pred)["bss"]
        print(f"  valid={valid_season}  BSS={bss:.6f}  score={max(0,bss*100000):.1f}  "
              f"best_iter={m.get_best_iteration()}  ({time.time()-tm:.0f}s)", flush=True)

    print("\n===== 전체 데이터(2019-2024) 최종 학습 =====", flush=True)
    tm = time.time()
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False, extra_features=EXTRA_FEATURES,
                        include_team_te=False).fit(df)
    X_full = fb.transform_train_oof(df)
    X_full = X_full.drop(columns=[c for c in DEAD_LIST_EXCL_SEASON if c in X_full.columns])
    X_full["pitcher_id"] = df["pitcher_id"].to_numpy()
    X_full["batter_id"] = df["batter_id"].to_numpy()
    cat_features = [c for c in CAT_FEATURES_BASE if c in X_full.columns] + ID_COLS
    for c in cat_features:
        X_full[c] = X_full[c].astype(np.int64).astype(str)
    y_full = df[TARGET_COL].to_numpy()

    tr_idx, es_idx = time_split_es(len(X_full))
    pool_tr = Pool(X_full.iloc[tr_idx], y_full[tr_idx].astype(np.float64), cat_features=cat_features)
    pool_es = Pool(X_full.iloc[es_idx], y_full[es_idx].astype(np.float64), cat_features=cat_features)
    model_full = CatBoostRegressor(loss_function="RMSE", **CB_PARAMS)
    model_full.fit(pool_tr, eval_set=pool_es)
    print(f"  최종 학습 완료 best_iter={model_full.get_best_iteration()} ({time.time()-tm:.0f}s)", flush=True)

    os.makedirs("phase4_preds", exist_ok=True)
    joblib.dump({
        "model": model_full, "stats": fb.export_stats(), "cat_features": cat_features,
        "drop_cols": DEAD_LIST_EXCL_SEASON, "extra_features": EXTRA_FEATURES,
    }, "phase4_preds/catboost_fixed_full.pkl")
    print(f"저장: phase4_preds/catboost_fixed_full.pkl  총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
