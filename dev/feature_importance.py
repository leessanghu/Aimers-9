"""Permutation feature importance on the 2024 validation split.

The score is Brier Skill Score, but the ranking uses Brier degradation:
larger positive delta means the ensemble gets worse when that feature is
shuffled, so the feature is more useful.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from metrics import evaluate, format_report

DATA_PATH = "../data/train.csv"
SEED = 42
W_RF = 0.15
N_SAMPLE = 60000
N_REPEATS = 2

RF_PARAMS = dict(
    n_estimators=300,
    max_depth=10,
    min_samples_leaf=200,
    n_jobs=-1,
    random_state=SEED,
)
HGB_PARAMS = dict(
    max_depth=6,
    max_leaf_nodes=31,
    max_iter=500,
    learning_rate=0.03,
    l2_regularization=5.0,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=SEED,
)


def predict_ensemble(rf, hgb, x):
    return W_RF * rf.predict_proba(x)[:, 1] + (1 - W_RF) * hgb.predict_proba(x)[:, 1]


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    train_fold = df[df["season"] <= 2023].reset_index(drop=True)
    valid_fold = df[df["season"] == 2024].reset_index(drop=True)
    y_train = train_fold[TARGET_COL].to_numpy()
    y_valid_full = valid_fold[TARGET_COL].to_numpy()

    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(train_fold)
    x_train = fb.transform_train_oof(train_fold)
    x_valid_full = fb.transform(valid_fold)

    rf = RandomForestClassifier(**RF_PARAMS).fit(x_train, y_train)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(x_train, y_train)

    if N_SAMPLE and len(x_valid_full) > N_SAMPLE:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(x_valid_full), size=N_SAMPLE, replace=False)
        x_valid = x_valid_full.iloc[idx].reset_index(drop=True)
        y_valid = y_valid_full[idx]
    else:
        x_valid = x_valid_full.reset_index(drop=True)
        y_valid = y_valid_full

    base_pred = predict_ensemble(rf, hgb, x_valid)
    base_metrics = evaluate(y_valid, base_pred)
    base_bs = base_metrics["brier_score"]
    print(format_report("base ensemble sample", base_metrics))
    print(f"sample={len(x_valid):,}  features={x_valid.shape[1]}  repeats={N_REPEATS}")

    rng = np.random.default_rng(SEED + 1)
    rows = []
    for feature in x_valid.columns:
        deltas = []
        for _ in range(N_REPEATS):
            x_perm = x_valid.copy()
            x_perm[feature] = rng.permutation(x_perm[feature].to_numpy())
            pred = predict_ensemble(rf, hgb, x_perm)
            deltas.append(evaluate(y_valid, pred)["brier_score"] - base_bs)
        rows.append(
            {
                "feature": feature,
                "brier_delta_mean": float(np.mean(deltas)),
                "brier_delta_std": float(np.std(deltas)),
            }
        )

    result = pd.DataFrame(rows).sort_values("brier_delta_mean", ascending=False)
    out_path = "feature_importance_2024.csv"
    result.to_csv(out_path, index=False, encoding="utf-8")

    print("\nTop 30 permutation importance")
    print(result.head(30).to_string(index=False))
    print(f"\nSaved {out_path}  elapsed={time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
