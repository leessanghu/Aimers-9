"""여러 시계열 폴드에서 공통으로 '죽은' 피처를 보수적으로 찾는다.

폴드: 2019-2021->2022, 2019-2022->2023, 2019-2023->2024 (전체 train 범위를 3번 훑음)
각 폴드에서 LGBM classifier로 permutation importance(Brier 악화량)를 재고,
"모든 폴드에서" 악화량이 임계값(THRESH) 이하인 피처만 최종 후보로 채택한다.
(한 폴드에서만 약해 보이는 피처는 노이즈일 수 있어 보수적으로 제외하지 않는다.)
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from features import FeatureBuilder, TARGET_COL
from metrics import evaluate

DATA_PATH = "../data/train.csv"
SEED = 42
FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]  # (train<=X, valid==X+1)
N_REPEATS = 3
THRESH = 0.00002  # 이 값 이하 악화량이면 '이 폴드에서는 죽은 피처' 취급

LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)

CAT_FEATURES = ["cat_top_bottom", "cat_game_type", "cat_base_state", "count_state", "hand_matchup"]


def fold_importance(df, train_max_season, valid_season):
    train_fold = df[df["season"] <= train_max_season].reset_index(drop=True)
    valid_fold = df[df["season"] == valid_season].reset_index(drop=True)
    y_train = train_fold[TARGET_COL].to_numpy()
    y_valid = valid_fold[TARGET_COL].to_numpy()

    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(train_fold)
    X_train = fb.transform_train_oof(train_fold)
    X_valid = fb.transform(valid_fold)
    for c in CAT_FEATURES:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype(pd.CategoricalDtype(categories=X_train[c].cat.categories))

    cut = int(len(X_train) * 0.92)
    tr_idx, es_idx = np.arange(cut), np.arange(cut, len(X_train))

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train.iloc[tr_idx], y_train[tr_idx],
             eval_set=[(X_train.iloc[es_idx], y_train[es_idx])], eval_metric="binary_logloss",
             callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])

    base_pred = model.predict_proba(X_valid)[:, 1]
    base_bs = evaluate(y_valid, base_pred)["brier_score"]

    rng = np.random.default_rng(SEED + train_max_season)
    rows = []
    for feature in X_valid.columns:
        deltas = []
        orig_dtype = X_valid[feature].dtype
        for _ in range(N_REPEATS):
            X_perm = X_valid.copy()
            permuted = rng.permutation(X_perm[feature].to_numpy())
            X_perm[feature] = pd.Series(permuted, index=X_perm.index).astype(orig_dtype)
            p = model.predict_proba(X_perm)[:, 1]
            deltas.append(evaluate(y_valid, p)["brier_score"] - base_bs)
        rows.append({"feature": feature, "delta": float(np.mean(deltas))})
    return pd.DataFrame(rows).set_index("feature")["delta"]


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    all_deltas = {}
    for train_max, valid_s in FOLDS:
        print(f"[fold] train<=season{train_max} -> valid=season{valid_s} 학습+permutation...", flush=True)
        tf = time.time()
        all_deltas[valid_s] = fold_importance(df, train_max, valid_s)
        print(f"  완료 ({time.time()-tf:.0f}s)", flush=True)

    table = pd.DataFrame(all_deltas)
    table["min_delta"] = table.min(axis=1)
    table["max_delta"] = table.max(axis=1)
    table = table.sort_values("min_delta")

    dead = table[(table[[2022, 2023, 2024]] <= THRESH).all(axis=1)]
    print(f"\n임계값 {THRESH} 기준, 3개 폴드 전부에서 죽은 피처 ({len(dead)}개):", flush=True)
    print(dead.to_string(), flush=True)

    table.to_csv("dead_features_all_folds.csv", encoding="utf-8")
    dead.index.to_series().to_csv("dead_features_conservative_list.csv", index=False, header=["feature"])
    print(f"\n저장: dead_features_all_folds.csv, dead_features_conservative_list.csv", flush=True)
    print(f"총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
