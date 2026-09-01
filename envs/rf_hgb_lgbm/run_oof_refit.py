from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from modeling import DEV, ROOT, fit_lgbm_l2_full

import sys
if str(DEV) not in sys.path:
    sys.path.insert(0, str(DEV))

from metrics import evaluate  # noqa: E402
from phase2_common import FOLDS, build_fold  # noqa: E402

OUT = Path(__file__).resolve().parent / "oof"


def main():
    OUT.mkdir(exist_ok=True)
    df = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    summary = []

    for train_max, valid_season in FOLDS:
        started = time.time()
        print(f"\n===== train<={train_max} -> valid={valid_season} =====", flush=True)
        fold = build_fold(df, train_max, valid_season, extra_features=None, seed=42, include_team_te=True)
        X_train, X_valid = fold["X_train"], fold["X_valid"]
        y_train, y_valid = fold["y_train"], fold["y_valid"]

        # RF/HGB/classifier OOF was already produced with exactly this fold and
        # feature definition. Reuse it so concurrent CatBoost work does not make
        # us spend most of the experiment budget reproducing identical trees.
        baseline_path = DEV / "phase2_preds" / f"fold_{valid_season}_preds.csv"
        baseline = pd.read_csv(baseline_path)
        if not baseline["row_id"].equals(pd.Series(fold["row_id"])):
            raise ValueError(f"row_id mismatch: {baseline_path}")
        if not baseline["y_valid"].equals(pd.Series(y_valid)):
            raise ValueError(f"target mismatch: {baseline_path}")

        lgbm_l2, l2_iter = fit_lgbm_l2_full(X_train, y_train)
        print(f"  LGBM L2 complete, iterations={l2_iter}", flush=True)

        preds = {
            "rf": baseline["pred_rf"].to_numpy(),
            "hgb": baseline["pred_hgb"].to_numpy(),
            "lgbm_cls": baseline["pred_lgbm_cls"].to_numpy(),
            "lgbm_l2": lgbm_l2.predict(X_valid).clip(0.0, 1.0),
        }
        out = pd.DataFrame({"row_id": fold["row_id"], "y_valid": y_valid})
        for name, pred in preds.items():
            out[f"pred_{name}"] = pred
            result = evaluate(y_valid, pred)
            summary.append({
                "valid_season": valid_season,
                "model": name,
                "bss": result["bss"],
                "brier": result["brier_score"],
                "pred_mean": float(pred.mean()),
                "target_mean": float(y_valid.mean()),
                "iterations": {"lgbm_l2": l2_iter}.get(name),
            })
            print(f"  {name}: BSS={result['bss']:.6f}, pred_mean={pred.mean():.6f}", flush=True)

        out.to_csv(OUT / f"fold_{valid_season}.csv", index=False)
        print(f"  fold seconds={time.time() - started:.1f}", flush=True)

    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    metadata = {
        "feature_config": "full + team_te",
        "rf_hgb_lgbm_cls_source": "dev/phase2_preds",
        "lgbm_l2_refit_after_early_stopping": True,
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
