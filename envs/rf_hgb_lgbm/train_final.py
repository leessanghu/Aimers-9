from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

from modeling import DEV, HGB_PARAMS, ROOT, fit_lgbm_l2_full, fit_rf_full

if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from features import FeatureBuilder, TARGET_COL  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "submission" / "model"


def main():
    started = time.time()
    selected = json.loads((HERE / "selected_blend.json").read_text(encoding="utf-8"))
    df = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    y = df[TARGET_COL].to_numpy()
    fb = FeatureBuilder(seed=42, include_raw_rates=False, extra_features=None, include_team_te=True).fit(df)
    X = fb.transform_train_oof(df)

    rf = fit_rf_full(X, y)
    print("RF full refit complete", flush=True)
    # Match the HGB procedure used by the reused Phase 2 OOF exactly.
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
    hgb_iter = int(hgb.n_iter_)
    print(f"HGB complete, iterations={hgb_iter}", flush=True)
    lgbm_l2, l2_iter = fit_lgbm_l2_full(X, y)
    print(f"LGBM L2 full refit complete, iterations={l2_iter}", flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    artifact = {
        "stats": fb.export_stats(),
        "models": {"rf": rf, "hgb": hgb, "lgbm_l2": lgbm_l2},
        "weights": selected["weights"],
        "iterations": {"hgb": hgb_iter, "lgbm_l2": l2_iter},
        "oof_metrics": selected["metrics"],
    }
    joblib.dump(artifact, OUT / "model_artifacts.pkl", compress=3)
    print(f"saved in {time.time() - started:.1f}s: {OUT / 'model_artifacts.pkl'}")


if __name__ == "__main__":
    main()
