"""Final submit model v5: v4 RF/HGB + recent pitcher success windows."""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from inseason import build_season_end_table, transform_inseason, export_stats as inseason_export_stats
from recent_pitcher import build_recent_success_features

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42

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

_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")


def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG_TYPES:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strip_rng(v, seen, depth + 1)


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    y = df[TARGET_COL].to_numpy()
    print(f"train={len(df):,} season={df.season.min()}~{df.season.max()}", flush=True)

    print("\n[1] FeatureBuilder base", flush=True)
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(df)
    X_base = fb.transform_train_oof(df).reset_index(drop=True)
    print(f"  base={X_base.shape[1]} ({time.time() - t0:.0f}s)", flush=True)

    print("\n[2] in-season v4 features", flush=True)
    season_end_table = build_season_end_table(df)
    seasons_range = sorted(df["season"].unique().tolist())
    global_success_rate = float(df[TARGET_COL].mean())
    X_inseason = transform_inseason(df, season_end_table, global_success_rate, seasons_range).reset_index(drop=True)
    print(f"  inseason={X_inseason.shape[1]} ({time.time() - t0:.0f}s)", flush=True)

    print("\n[3] recent success windows", flush=True)
    X_recent = build_recent_success_features(
        df, X_inseason["inseason_success_smooth"].to_numpy(),
        windows=(50, 100, 200), k_smooth=25.0)[RECENT_COLS]
    print(f"  recent={X_recent.shape[1]} ({time.time() - t0:.0f}s)", flush=True)

    X = pd.concat([X_base, X_inseason, X_recent], axis=1)
    print(f"\nfeatures={X.shape[1]}", flush=True)

    print("\n[4] RF fit", flush=True)
    rf = RandomForestClassifier(**RF_PARAMS).fit(X, y)
    print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    print("\n[5] HGB fit", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
    print(f"  done ({time.time() - t0:.0f}s)", flush=True)

    strip_rng(rf)
    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    artifacts = {
        "rf": rf,
        "hgb": hgb,
        "w_rf": W_RF,
        "w_hgb": W_HGB,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(season_end_table, global_success_rate, seasons_range),
        "recent_cols": RECENT_COLS,
        "recent_windows": [50, 100, 200],
        "recent_k_smooth": 25.0,
    }
    out_path = os.path.join(OUT_DIR, "model_artifacts_v5.pkl")
    joblib.dump(artifacts, out_path, compress=3)
    print(f"\nsaved {out_path} size={os.path.getsize(out_path)/1e6:.1f}MB total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
