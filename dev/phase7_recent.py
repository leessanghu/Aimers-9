"""Phase 7: test recent pitcher-state features with the submit v4 model family.

Keeps the current submit-side algorithm choice intact:
  - RandomForestClassifier
  - HistGradientBoostingClassifier
  - fixed RF/HGB blend

LightGBM L2 is included only as a diagnostic because prior experiments showed
it is a strong tabular baseline, but it is not part of submit/model_artifacts_v4.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from inseason import build_season_end_table, transform_inseason
from metrics import evaluate
from phase2_common import time_split_es
from recent_pitcher import build_recent_success_features

SEED = 42
DATA_PATH = "../data/train.csv"

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                  l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=SEED)
LGBM_PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
                   min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
                   colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)
W_RF, W_HGB = 0.15, 0.85


def fit_predict(X_train, y_train, X_valid):
    preds = {}

    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    preds["rf"] = rf.predict_proba(X_valid)[:, 1]

    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)
    preds["hgb"] = hgb.predict_proba(X_valid)[:, 1]
    preds["rf015_hgb085"] = W_RF * preds["rf"] + W_HGB * preds["hgb"]

    tr_idx, es_idx = time_split_es(len(X_train))
    lgb = LGBMRegressor(**LGBM_PARAMS)
    lgb.fit(X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64),
            eval_set=[(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64))],
            eval_metric="l2",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    preds["lgbm_l2_diag"] = np.clip(lgb.predict(X_valid), 0.0, 1.0)
    return preds


def score_predictions(y_valid, preds, prefix):
    rows = []
    for model, pred in preds.items():
        m = evaluate(y_valid, pred)
        rows.append({
            "variant": prefix,
            "model": model,
            "bss": m["bss"],
            "score": m["leaderboard_score"],
            "brier": m["brier_score"],
            "target_mean": float(y_valid.mean()),
            "pred_mean": float(np.mean(pred)),
        })
        print(f"  {model:15s} BSS={m['bss']:.6f} score={m['leaderboard_score']:.1f} "
              f"pred_mean={np.mean(pred):.6f}", flush=True)
    return rows


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    print(f"load train={len(df):,} season={df.season.min()}~{df.season.max()}", flush=True)

    train_mask = df["season"] <= 2023
    valid_mask = df["season"] == 2024
    train_df = df.loc[train_mask].reset_index(drop=True)
    valid_df = df.loc[valid_mask].reset_index(drop=True)
    y_train = train_df[TARGET_COL].to_numpy()
    y_valid = valid_df[TARGET_COL].to_numpy()

    print("\n[1] base FeatureBuilder fit/transform", flush=True)
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(train_df)
    X_train_base = fb.transform_train_oof(train_df).reset_index(drop=True)
    X_valid_base = fb.transform(valid_df).reset_index(drop=True)
    print(f"  base={X_train_base.shape[1]} features ({time.time() - t0:.0f}s)", flush=True)

    print("\n[2] current submit in-season features", flush=True)
    season_end = build_season_end_table(train_df)
    seasons_range = sorted(train_df["season"].unique().tolist())
    global_success_rate = float(train_df[TARGET_COL].mean())
    X_train_ins = transform_inseason(train_df, season_end, global_success_rate, seasons_range).reset_index(drop=True)
    X_valid_ins = transform_inseason(valid_df, season_end, global_success_rate, seasons_range).reset_index(drop=True)
    current_cols = list(X_train_ins.columns)
    print(f"  inseason={len(current_cols)} features ({time.time() - t0:.0f}s)", flush=True)

    print("\n[3] recent success features from asof prefix counts", flush=True)
    X_train_recent = build_recent_success_features(
        train_df, X_train_ins["inseason_success_smooth"].to_numpy(), windows=(50, 100, 200), k_smooth=25.0)
    X_valid_recent = build_recent_success_features(
        valid_df, X_valid_ins["inseason_success_smooth"].to_numpy(), windows=(50, 100, 200), k_smooth=25.0)
    keep_recent = [
        "recent_success_50_smooth",
        "recent_success_100_smooth",
        "recent_success_200_smooth",
        "recent_success_50_minus_inseason",
        "recent_success_100_minus_inseason",
        "recent_success_200_minus_inseason",
        "recent_success_50_logit_minus_inseason",
        "recent_success_200_logit_minus_inseason",
        "recent_success_50_n",
        "recent_success_200_n",
    ]
    X_train_recent = X_train_recent[keep_recent]
    X_valid_recent = X_valid_recent[keep_recent]
    print(f"  recent={X_train_recent.shape[1]} features ({time.time() - t0:.0f}s)", flush=True)
    print(X_valid_recent.describe().T[["mean", "std", "min", "max"]].round(6), flush=True)

    results = []

    print("\n===== v4_current: base + submit in-season 5 =====", flush=True)
    X_train_current = pd.concat([X_train_base, X_train_ins], axis=1)
    X_valid_current = pd.concat([X_valid_base, X_valid_ins], axis=1)
    preds_current = fit_predict(X_train_current, y_train, X_valid_current)
    results.extend(score_predictions(y_valid, preds_current, "v4_current"))

    print("\n===== v4_recent: current + recent success windows =====", flush=True)
    X_train_recent_full = pd.concat([X_train_current, X_train_recent], axis=1)
    X_valid_recent_full = pd.concat([X_valid_current, X_valid_recent], axis=1)
    preds_recent = fit_predict(X_train_recent_full, y_train, X_valid_recent_full)
    results.extend(score_predictions(y_valid, preds_recent, "v4_recent"))

    out = pd.DataFrame(results).sort_values(["model", "variant"])
    out_path = "phase7_recent_2024.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    print("\n===== delta: v4_recent - v4_current =====", flush=True)
    piv = out.pivot(index="model", columns="variant", values="bss")
    for model in piv.index:
        if {"v4_current", "v4_recent"} <= set(piv.columns):
            delta = piv.loc[model, "v4_recent"] - piv.loc[model, "v4_current"]
            print(f"  {model:15s} delta_bss={delta:+.6f} delta_score={delta * 100000:+.1f}", flush=True)
    print(f"\nsaved {out_path} total={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
