"""최종 제출 모델 v2: LGBM-D(0.6) + LGBM-A(0.4) 블렌딩, 2019~2024 전체 학습.

- A: 58피처(전체) + team_te 포함
- D: pruned 23피처(35개 제거, season은 유지) + count_asof_ball/diff_prev1_prev5 + team_te 제거
- 둘 다 phase2_optuna에서 튜닝된 LGBM L2(reg:squarederror -> clip) 파라미터 사용
- 3-fold(2022/2023/2024) grid search로 확인된 최적 블렌딩: w_D=0.6, w_A=0.4 (가중BSS 0.004676)
"""

import os
import time

import joblib
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from features import FeatureBuilder, TARGET_COL
from phase2_common import time_split_es

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42

TUNED_L2_PARAMS = dict(
    num_leaves=64, max_depth=12, learning_rate=0.005571638320335239,
    min_child_samples=28, subsample=0.9017762093981382,
    colsample_bytree=0.5291780969405919, reg_alpha=0.07089938907781941,
    reg_lambda=0.009306216375166584, min_split_gain=0.4888649495163153, max_bin=127,
    n_estimators=3000, subsample_freq=1, random_state=SEED, n_jobs=-1, verbosity=-1,
)

DEAD_LIST_EXCL_SEASON = [
    c for c in pd.read_csv("dead_features_conservative_list.csv")["feature"].tolist()
    if c != "season"
]

W_D = 0.6
W_A = 0.4


def fit_lgbm_l2(X, y):
    tr_idx, es_idx = time_split_es(len(X))
    m = LGBMRegressor(**TUNED_L2_PARAMS)
    m.fit(X.iloc[tr_idx], y[tr_idx].astype(np.float64),
         eval_set=[(X.iloc[es_idx], y[es_idx].astype(np.float64))], eval_metric="l2",
         callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    return m


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"전체 train={len(df):,}  season {df['season'].min()}~{df['season'].max()}")
    y = df[TARGET_COL].to_numpy()

    # ---- A: full 58 + team_te ----
    print("\n[A] full 피처 + team_te, FeatureBuilder fit...", flush=True)
    fb_a = FeatureBuilder(seed=SEED, include_raw_rates=False, extra_features=None,
                          include_team_te=True).fit(df)
    X_a = fb_a.transform_train_oof(df)
    print(f"  피처 수={X_a.shape[1]}  ({time.time()-t0:.0f}s)", flush=True)
    model_a = fit_lgbm_l2(X_a, y)
    print(f"  A 학습 완료 best_iter={model_a.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    # ---- D: pruned + 신규 2개, team_te 제거 ----
    print("\n[D] pruned 피처 + 신규2 + team_te 제거, FeatureBuilder fit...", flush=True)
    fb_d = FeatureBuilder(seed=SEED, include_raw_rates=False,
                          extra_features={"count_asof_ball", "diff_prev1_prev5"},
                          include_team_te=False).fit(df)
    X_d_full = fb_d.transform_train_oof(df)
    X_d = X_d_full.drop(columns=[c for c in DEAD_LIST_EXCL_SEASON if c in X_d_full.columns])
    print(f"  피처 수={X_d.shape[1]}  ({time.time()-t0:.0f}s)", flush=True)
    model_d = fit_lgbm_l2(X_d, y)
    print(f"  D 학습 완료 best_iter={model_d.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    artifacts = {
        "model_a": model_a, "stats_a": fb_a.export_stats(),
        "model_d": model_d, "stats_d": fb_d.export_stats(),
        "drop_cols_d": DEAD_LIST_EXCL_SEASON,
        "w_a": W_A, "w_d": W_D,
        "a_columns": list(X_a.columns), "d_columns": list(X_d.columns),
    }
    out_path = os.path.join(OUT_DIR, "model_artifacts_v2.pkl")
    joblib.dump(artifacts, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n저장 완료: {out_path} ({size_mb:.1f}MB)  총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
