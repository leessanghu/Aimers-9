"""최종 제출 모델 v4: RF+HGB(811점 구성, 58피처+team_te) + in-season 피처 5개.
2019~2024 전체 학습. numpy RNG(HGB) strip 포함(v1과 동일 안전장치)."""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from inseason import build_season_end_table, transform_inseason, export_stats as inseason_export_stats

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
W_RF, W_HGB = 0.15, 0.85

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
    print(f"전체 train={len(df):,}  season {df['season'].min()}~{df['season'].max()}")
    y = df[TARGET_COL].to_numpy()

    print("\n[1] 기존 FeatureBuilder (58피처 + team_te)...", flush=True)
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(df)
    X_base = fb.transform_train_oof(df)
    print(f"  피처 수={X_base.shape[1]}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[2] in-season 피처 (5개)...", flush=True)
    season_end_table = build_season_end_table(df)
    seasons_range = sorted(df["season"].unique().tolist())
    global_success_rate = float(df[TARGET_COL].mean())
    X_inseason = transform_inseason(df, season_end_table, global_success_rate, seasons_range)
    X_inseason = X_inseason.reset_index(drop=True)
    X_full = pd.concat([X_base.reset_index(drop=True), X_inseason], axis=1)
    print(f"  최종 피처 수={X_full.shape[1]}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[3] RF 학습...", flush=True)
    rf = RandomForestClassifier(**RF_PARAMS).fit(X_full, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_full, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(rf)
    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    artifacts = {
        "rf": rf, "hgb": hgb, "w_rf": W_RF, "w_hgb": W_HGB,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(season_end_table, global_success_rate, seasons_range),
    }
    out_path = os.path.join(OUT_DIR, "model_artifacts_v4.pkl")
    joblib.dump(artifacts, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"\n저장 완료: {out_path} ({size_mb:.1f}MB)  총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
