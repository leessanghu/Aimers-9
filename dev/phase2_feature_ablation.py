"""Phase 2 - 4: 신규 교차/차분/season-count 피처를 하나씩 추가해 ablation.

베이스: Phase 1 챔피언 LGBM classifier 설정(recency 없음).
평가: 3-fold(2022/2023/2024) 가중 BSS(0.2/0.3/0.5, equal, 0.1/0.2/0.7 민감도 포함).
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from features import EXTRA_FEATURE_NAMES, RISKY_EXTRA_FEATURES
from metrics import evaluate
from phase2_common import FOLDS, build_fold, load_full, time_split_es, weighted_bss, FOLD_WEIGHTS

SEED = 42
LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)


def run_config(df, extra_features, label):
    fold_bss = {}
    for train_max, valid_season in FOLDS:
        fold = build_fold(df, train_max, valid_season, extra_features=extra_features, seed=SEED)
        X_train, y_train = fold["X_train"], fold["y_train"]
        tr_idx, es_idx = time_split_es(len(X_train))
        m = LGBMClassifier(**LGBM_PARAMS)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
             eval_set=[(X_train.iloc[es_idx], y_train[es_idx])], eval_metric="binary_logloss",
             callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        p = m.predict_proba(fold["X_valid"])[:, 1]
        fold_bss[valid_season] = evaluate(fold["y_valid"], p)["bss"]
    wbss = {wname: weighted_bss(fold_bss, w) for wname, w in FOLD_WEIGHTS.items()}
    return {"config": label, **fold_bss, **wbss}


def main():
    t0 = time.time()
    df = load_full()

    rows = []
    print("baseline(추가 피처 없음) 평가...", flush=True)
    tb = time.time()
    rows.append(run_config(df, extra_features=None, label="baseline"))
    print(f"  {rows[-1]}  ({time.time()-tb:.0f}s)", flush=True)

    for feat in EXTRA_FEATURE_NAMES:
        tb = time.time()
        row = run_config(df, extra_features={feat}, label=f"+{feat}")
        row["risky"] = feat in RISKY_EXTRA_FEATURES
        rows.append(row)
        tag = "  [!] risky(train 내부 미래정보 섞임 — 신뢰 보류)" if row["risky"] else ""
        print(f"{feat}: {row}{tag}  ({time.time()-tb:.0f}s)", flush=True)

    result = pd.DataFrame(rows)
    result["risky"] = result["risky"].fillna(False)
    result = result.sort_values("default_0.2_0.3_0.5", ascending=False)
    result.to_csv("phase2_feature_ablation.csv", index=False, encoding="utf-8")
    print("\n===== 피처 ablation 결과 (가중 BSS 내림차순) =====", flush=True)
    print(result.to_string(index=False), flush=True)
    print("\n[!] risky=True (season_count 4종)는 last-train-season count를 train 전체 행에 "
          "동일 매핑해서 만든 값이라 train 내부에 미래 정보가 섞여 있음.", flush=True)
    print("    row별 previous-season/expanding count로 재구현하기 전까지 이 결과는 채택 근거로 쓰지 말 것.",
          flush=True)
    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
