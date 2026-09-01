"""Three-fold validation for recent pitcher success features.

This uses the current submit model family only: RF, HGB, and the fixed
RF0.15/HGB0.85 blend. It compares the deployed v4 feature set against v4 plus
recent success windows reconstructed from asof cumulative counts.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from inseason import build_season_end_table, transform_inseason
from metrics import evaluate
from recent_pitcher import build_recent_success_features

SEED = 42
DATA_PATH = "../data/train.csv"
FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]
FOLD_WEIGHTS = {2022: 0.2, 2023: 0.3, 2024: 0.5}

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                  l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=SEED)
W_RF, W_HGB = 0.15, 0.85

RECENT_COLS = [
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


def fit_preds(X_train, y_train, X_valid):
    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)
    p_rf = rf.predict_proba(X_valid)[:, 1]
    p_hgb = hgb.predict_proba(X_valid)[:, 1]
    return {
        "rf": p_rf,
        "hgb": p_hgb,
        "rf015_hgb085": W_RF * p_rf + W_HGB * p_hgb,
    }


def add_rows(rows, valid_season, variant, y_valid, preds):
    for model, pred in preds.items():
        m = evaluate(y_valid, pred)
        rows.append({
            "valid_season": valid_season,
            "variant": variant,
            "model": model,
            "bss": m["bss"],
            "score": m["leaderboard_score"],
            "brier": m["brier_score"],
            "target_mean": float(y_valid.mean()),
            "pred_mean": float(np.mean(pred)),
        })
        print(f"    {variant:10s} {model:13s} BSS={m['bss']:.6f} "
              f"score={m['leaderboard_score']:.1f} pred_mean={np.mean(pred):.6f}", flush=True)


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    rows = []

    for train_max, valid_season in FOLDS:
        print(f"\n===== fold train<={train_max} valid={valid_season} =====", flush=True)
        train_df = df[df["season"] <= train_max].reset_index(drop=True)
        valid_df = df[df["season"] == valid_season].reset_index(drop=True)
        y_train = train_df[TARGET_COL].to_numpy()
        y_valid = valid_df[TARGET_COL].to_numpy()

        fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(train_df)
        X_train_base = fb.transform_train_oof(train_df).reset_index(drop=True)
        X_valid_base = fb.transform(valid_df).reset_index(drop=True)

        season_end = build_season_end_table(train_df)
        seasons_range = sorted(train_df["season"].unique().tolist())
        global_success_rate = float(train_df[TARGET_COL].mean())
        X_train_ins = transform_inseason(train_df, season_end, global_success_rate, seasons_range).reset_index(drop=True)
        X_valid_ins = transform_inseason(valid_df, season_end, global_success_rate, seasons_range).reset_index(drop=True)

        X_train_current = pd.concat([X_train_base, X_train_ins], axis=1)
        X_valid_current = pd.concat([X_valid_base, X_valid_ins], axis=1)
        print(f"  current features={X_train_current.shape[1]}", flush=True)
        add_rows(rows, valid_season, "current", y_valid, fit_preds(X_train_current, y_train, X_valid_current))

        X_train_recent = build_recent_success_features(
            train_df, X_train_ins["inseason_success_smooth"].to_numpy(),
            windows=(50, 100, 200), k_smooth=25.0)[RECENT_COLS]
        X_valid_recent = build_recent_success_features(
            valid_df, X_valid_ins["inseason_success_smooth"].to_numpy(),
            windows=(50, 100, 200), k_smooth=25.0)[RECENT_COLS]
        X_train_full = pd.concat([X_train_current, X_train_recent], axis=1)
        X_valid_full = pd.concat([X_valid_current, X_valid_recent], axis=1)
        print(f"  recent features={X_train_full.shape[1]}", flush=True)
        add_rows(rows, valid_season, "recent", y_valid, fit_preds(X_train_full, y_train, X_valid_full))
        print(f"  elapsed={time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("phase7_recent_3fold.csv", index=False, encoding="utf-8-sig")

    print("\n===== weighted summary (0.2/0.3/0.5) =====", flush=True)
    for model in ["rf", "hgb", "rf015_hgb085"]:
        for variant in ["current", "recent"]:
            sub = out[(out["model"] == model) & (out["variant"] == variant)]
            wbss = sum(float(sub[sub["valid_season"] == s]["bss"].iloc[0]) * w
                       for s, w in FOLD_WEIGHTS.items())
            print(f"  {variant:10s} {model:13s} weighted_bss={wbss:.6f} score={wbss*100000:.1f}", flush=True)
        cur = out[(out["model"] == model) & (out["variant"] == "current")]
        rec = out[(out["model"] == model) & (out["variant"] == "recent")]
        d = sum((float(rec[rec["valid_season"] == s]["bss"].iloc[0]) -
                 float(cur[cur["valid_season"] == s]["bss"].iloc[0])) * w
                for s, w in FOLD_WEIGHTS.items())
        print(f"  delta      {model:13s} weighted_delta={d:+.6f} score={d*100000:+.1f}", flush=True)

    print(f"\nsaved phase7_recent_3fold.csv total={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
