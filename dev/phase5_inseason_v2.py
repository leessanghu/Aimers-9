"""in-season v2 검증 — 2024 폴드.

테스트 구성:
  baseline      : 58피처만
  v1            : 기존 5개 (ball/reverse만, K=15 사전 스무딩)
  v2_raw        : raw 성분(success/ball/reverse/middle/strike) + n + cold-start, 사전 스무딩 없음
  v2_raw_trend  : v2_raw + 시즌 간 추세 + 통산(직전시즌까지) 수준
  k_sweep       : success rate 단독 스무딩 K in {15,30,50,100} 각각 baseline에 추가 (K 민감도 격리 측정)
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from inseason_v2 import build_season_end_table, build_global_rates, transform_inseason_v2
from metrics import evaluate
from phase2_common import time_split_es

SEED = 42
RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
LGBM_PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
                    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
                    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)


def run_eval(X_train, y_train, X_valid, y_valid, tag):
    rows = {}
    tr_idx, es_idx = time_split_es(len(X_train))

    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    p_rf = rf.predict_proba(X_valid)[:, 1]
    rows["rf"] = evaluate(y_valid, p_rf)["bss"]

    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)
    p_hgb = hgb.predict_proba(X_valid)[:, 1]
    rows["hgb"] = evaluate(y_valid, p_hgb)["bss"]
    rows["rf015_hgb085"] = evaluate(y_valid, 0.15 * p_rf + 0.85 * p_hgb)["bss"]

    lgb = LGBMRegressor(**LGBM_PARAMS)
    lgb.fit(X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64),
           eval_set=[(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64))], eval_metric="l2",
           callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    rows["lgbm_a"] = evaluate(y_valid, np.clip(lgb.predict(X_valid), 0.0, 1.0))["bss"]

    print(f"[{tag}]")
    for k, v in rows.items():
        print(f"  {k:15s} BSS={v:.6f}  score={max(0,v*100000):.1f}")
    return rows


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)

    print("season_end_table 구성...", flush=True)
    season_end = build_season_end_table(df)
    global_rates = build_global_rates(df)
    seasons_range = sorted(df["season"].unique().tolist())
    print(f"  완료 ({time.time()-t0:.0f}s)  global_rates={ {k: round(v,3) for k,v in global_rates.items()} }", flush=True)

    v2_full = transform_inseason_v2(df, season_end, global_rates, seasons_range, k_smooth_list=(15, 30, 50, 100))
    print(f"  v2 피처 계산 완료 ({time.time()-t0:.0f}s), 컬럼: {list(v2_full.columns)}", flush=True)

    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(df[df["season"] <= 2023])
    X_base_full = fb.transform_train_oof(df[df["season"] <= 2023]).reset_index(drop=True)
    X_base_valid = fb.transform(df[df["season"] == 2024]).reset_index(drop=True)

    train_idx = df.index[df["season"] <= 2023]
    valid_idx = df.index[df["season"] == 2024]
    y_train = df.loc[train_idx, TARGET_COL].to_numpy()
    y_valid = df.loc[valid_idx, TARGET_COL].to_numpy()

    v2_train = v2_full.loc[train_idx].reset_index(drop=True)
    v2_valid = v2_full.loc[valid_idx].reset_index(drop=True)

    results = {}

    print("\n===== baseline =====", flush=True)
    results["baseline"] = run_eval(X_base_full, y_train, X_base_valid, y_valid, "baseline")

    v1_cols = ["inseason_ball_smooth_k15", "inseason_reverse_smooth_k15", "inseason_n",
               "inseason_is_first_appearance"]
    # v1은 success도 K=15 스무딩(구버전과 동일 이름은 아니지만 값은 같음) 포함해야 하므로 success_smooth_k15 추가
    v1_cols_full = ["inseason_success_smooth_k15"] + v1_cols
    print("\n===== v1 (기존 5개, K=15) =====", flush=True)
    results["v1"] = run_eval(pd.concat([X_base_full, v2_train[v1_cols_full]], axis=1), y_train,
                             pd.concat([X_base_valid, v2_valid[v1_cols_full]], axis=1), y_valid, "v1")

    raw_cols = [f"inseason_{k}_raw" for k in ["success", "ball", "reverse", "middle", "strike"]] + \
              ["inseason_n", "inseason_is_first_appearance"]
    print("\n===== v2_raw (A+C: raw 성분, 사전스무딩 없음) =====", flush=True)
    results["v2_raw"] = run_eval(pd.concat([X_base_full, v2_train[raw_cols]], axis=1), y_train,
                                 pd.concat([X_base_valid, v2_valid[raw_cols]], axis=1), y_valid, "v2_raw")

    trend_cols = raw_cols + ["season_trend_success", "prior_season_success_rate"]
    print("\n===== v2_raw_trend (A+B+C) =====", flush=True)
    results["v2_raw_trend"] = run_eval(pd.concat([X_base_full, v2_train[trend_cols]], axis=1), y_train,
                                       pd.concat([X_base_valid, v2_valid[trend_cols]], axis=1), y_valid,
                                       "v2_raw_trend")

    for k in [15, 30, 50, 100]:
        cols = [f"inseason_success_smooth_k{k}", "inseason_n", "inseason_is_first_appearance"]
        print(f"\n===== k_sweep success-only K={k} =====", flush=True)
        results[f"k{k}_success_only"] = run_eval(
            pd.concat([X_base_full, v2_train[cols]], axis=1), y_train,
            pd.concat([X_base_valid, v2_valid[cols]], axis=1), y_valid, f"k{k}_success_only")

    print("\n===== 전체 비교 (rf015_hgb085 기준) =====", flush=True)
    base_bss = results["baseline"]["rf015_hgb085"]
    for name, r in results.items():
        d = r["rf015_hgb085"] - base_bss
        print(f"  {name:22s}  rf015_hgb085={r['rf015_hgb085']:.6f}  lgbm_a={r['lgbm_a']:.6f}  "
              f"delta_vs_baseline={d:+.6f} ({d*100000:+.1f}점)")

    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
