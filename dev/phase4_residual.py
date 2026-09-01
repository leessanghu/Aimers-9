"""Phase 4-4: prior + residual 구조.

p0 = asof_pitcher_success_rate_smooth (시점 안전, 투수의 누적 제구 성공률)
residual target = y - p0
GBDT는 residual만 예측 -> final = clip(p0 + pred_residual, 0, 1)

동기: 가장 강한 단일 피처가 투수 과거 성공률이고, 연도별 calibration이 계속 움직인다.
절대 타깃을 직접 예측하는 대신 "현재 투수 상태에서 얼마나 위아래로 벗어나는가"를 학습하면
season drift에 대한 의존도를 줄일 수 있을 것이라는 가설.

평가 기준: 2024 단일 폴드 BSS를 1차로, 2022/2023은 참고 가드레일.
LGBM(D 피처셋)과 XGBoost(D 피처셋) 둘 다 residual-target으로 비교.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import xgboost as xgb
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from metrics import evaluate
from phase2_common import FOLDS, build_fold, load_full, time_split_es

SEED = 42
EXTRA_FEATURES = {"count_asof_ball", "diff_prev1_prev5"}
DEAD_LIST_EXCL_SEASON = [
    c for c in pd.read_csv("dead_features_conservative_list.csv")["feature"].tolist()
    if c != "season"
]
P0_COL = "asof_pitcher_success_rate_smooth"

LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)

XGB_PARAMS = dict(
    n_estimators=3000, learning_rate=0.01, max_depth=8, min_child_weight=20,
    subsample=0.9, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
    max_bin=256, tree_method="hist", enable_categorical=True,
    random_state=SEED, n_jobs=-1, early_stopping_rounds=100,
)


def prep_X(fold):
    X_train = fold["X_train"].drop(columns=[c for c in DEAD_LIST_EXCL_SEASON
                                            if c in fold["X_train"].columns]).copy()
    X_valid = fold["X_valid"].drop(columns=[c for c in DEAD_LIST_EXCL_SEASON
                                            if c in fold["X_valid"].columns]).copy()
    for c in X_train.columns:
        if str(X_train[c].dtype) == "category":
            tr_int = X_train[c].astype(np.float64).fillna(-1).astype(np.int64)
            va_int = X_valid[c].astype(np.float64).fillna(-1).astype(np.int64)
            cats = sorted(set(tr_int.unique()) | set(va_int.unique()))
            X_train[c] = tr_int.astype(pd.CategoricalDtype(categories=cats))
            X_valid[c] = va_int.astype(pd.CategoricalDtype(categories=cats))
    return X_train, X_valid


def run_lgbm_residual(fold):
    X_train, X_valid = prep_X(fold)
    y_train, y_valid = fold["y_train"], fold["y_valid"]
    p0_train, p0_valid = X_train[P0_COL].to_numpy(), X_valid[P0_COL].to_numpy()
    r_train = y_train - p0_train

    tr_idx, es_idx = time_split_es(len(X_train))
    m = LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_train.iloc[tr_idx], r_train[tr_idx],
         eval_set=[(X_train.iloc[es_idx], r_train[es_idx])], eval_metric="l2",
         callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    pred_r = m.predict(X_valid)
    pred = np.clip(p0_valid + pred_r, 0.0, 1.0)
    return evaluate(y_valid, pred)["bss"]


def run_xgb_residual(fold):
    X_train, X_valid = prep_X(fold)
    y_train, y_valid = fold["y_train"], fold["y_valid"]
    p0_train, p0_valid = X_train[P0_COL].to_numpy(), X_valid[P0_COL].to_numpy()
    r_train = (y_train - p0_train).astype(np.float64)

    tr_idx, es_idx = time_split_es(len(X_train))
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(X_train.iloc[tr_idx], r_train[tr_idx],
         eval_set=[(X_train.iloc[es_idx], r_train[es_idx])], verbose=False)
    pred_r = m.predict(X_valid)
    pred = np.clip(p0_valid + pred_r, 0.0, 1.0)
    return evaluate(y_valid, pred)["bss"]


def run_lgbm_direct(fold):
    """대조군: 같은 피처셋으로 residual 없이 y를 직접 예측."""
    X_train, X_valid = prep_X(fold)
    y_train, y_valid = fold["y_train"], fold["y_valid"]
    tr_idx, es_idx = time_split_es(len(X_train))
    m = LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64),
         eval_set=[(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64))], eval_metric="l2",
         callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    pred = np.clip(m.predict(X_valid), 0.0, 1.0)
    return evaluate(y_valid, pred)["bss"]


def main():
    t0 = time.time()
    df = load_full()

    rows = []
    for train_max, valid_season in FOLDS:
        print(f"\n===== fold: train<=season{train_max} -> valid=season{valid_season} =====", flush=True)
        fold = build_fold(df, train_max, valid_season, extra_features=EXTRA_FEATURES, seed=SEED,
                          include_team_te=False)

        tm = time.time()
        bss_direct = run_lgbm_direct(fold)
        print(f"  lgbm_direct(대조군)   BSS={bss_direct:.6f}  ({time.time()-tm:.0f}s)", flush=True)

        tm = time.time()
        bss_lgbm_res = run_lgbm_residual(fold)
        print(f"  lgbm_residual         BSS={bss_lgbm_res:.6f}  ({time.time()-tm:.0f}s)", flush=True)

        tm = time.time()
        bss_xgb_res = run_xgb_residual(fold)
        print(f"  xgb_residual          BSS={bss_xgb_res:.6f}  ({time.time()-tm:.0f}s)", flush=True)

        rows.append({"valid_season": valid_season, "lgbm_direct": bss_direct,
                    "lgbm_residual": bss_lgbm_res, "xgb_residual": bss_xgb_res})

    result = pd.DataFrame(rows)
    result.to_csv("phase4_residual_results.csv", index=False, encoding="utf-8")
    print("\n===== 결과 (2024이 1차 기준) =====", flush=True)
    print(result.to_string(index=False), flush=True)
    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
