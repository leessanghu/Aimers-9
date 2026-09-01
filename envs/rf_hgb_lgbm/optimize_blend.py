from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OOF = HERE / "oof"
SEASONS = (2022, 2023, 2024)
MODELS = ("rf", "hgb", "lgbm_cls", "lgbm_l2")
FOLD_WEIGHT_SETS = {
    "default": {2022: 0.2, 2023: 0.3, 2024: 0.5},
    "recent": {2022: 0.1, 2023: 0.2, 2024: 0.7},
    "equal": {2022: 1 / 3, 2023: 1 / 3, 2024: 1 / 3},
}


def bss(y, p):
    ref = np.mean((y - y.mean()) ** 2)
    return 1.0 - np.mean((y - p) ** 2) / ref


def weight_grid(step=0.025):
    units = int(round(1 / step))
    for parts in itertools.product(range(units + 1), repeat=len(MODELS) - 1):
        used = sum(parts)
        if used > units:
            continue
        full = (*parts, units - used)
        w = np.asarray(full, dtype=float) / units
        # Preserve the proven RF/HGB anchor and cap any one LGBM representation.
        if w[0] + w[1] < 0.55 or w[0] > 0.35 or w[2] + w[3] > 0.45:
            continue
        yield w


def main():
    folds = {s: pd.read_csv(OOF / f"fold_{s}.csv") for s in SEASONS}
    # A convex blend's Brier score is w.T @ E[e e.T] @ w. Precomputing this
    # 4x4 matrix makes the exhaustive grid exact without repeatedly scanning
    # roughly 750k OOF rows.
    fold_quadratics = {}
    for season, df in folds.items():
        y = df["y_valid"].to_numpy()
        matrix = np.column_stack([df[f"pred_{m}"].to_numpy() for m in MODELS])
        errors = matrix - y[:, None]
        fold_quadratics[season] = {
            "error_cross": errors.T @ errors / len(y),
            "reference": np.mean((y - y.mean()) ** 2),
        }
    rows = []
    for w in weight_grid():
        fold_bss = {}
        for season, stats in fold_quadratics.items():
            blend_brier = float(w @ stats["error_cross"] @ w)
            fold_bss[season] = 1.0 - blend_brier / stats["reference"]
        row = {f"w_{m}": w[i] for i, m in enumerate(MODELS)}
        row.update({f"bss_{s}": fold_bss[s] for s in SEASONS})
        for label, fold_weights in FOLD_WEIGHT_SETS.items():
            row[label] = sum(fold_bss[s] * fold_weights[s] for s in SEASONS)
        # Selection emphasizes the latest fold, but rejects gains bought by a
        # material collapse under the standard three-fold weighting.
        row["selection"] = 0.7 * row["recent"] + 0.3 * row["default"]
        rows.append(row)

    result = pd.DataFrame(rows)
    anchor = result[(result.w_rf == 0.15) & (result.w_hgb == 0.85)]
    anchor_2024 = float(anchor.iloc[0].bss_2024) if len(anchor) else 0.0
    eligible = result[result.bss_2024 >= anchor_2024].copy()
    eligible = eligible.sort_values(["recent", "default", "bss_2024"], ascending=False)
    result.sort_values("selection", ascending=False).to_csv(HERE / "blend_grid.csv", index=False)
    eligible.head(30).to_csv(HERE / "blend_shortlist.csv", index=False)

    best = eligible.iloc[0]
    selected = {
        "weights": {m: float(best[f"w_{m}"]) for m in MODELS},
        "metrics": {k: float(best[k]) for k in ["bss_2022", "bss_2023", "bss_2024", "default", "recent", "equal"]},
        "anchor_2024": anchor_2024,
    }
    (HERE / "selected_blend.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    print(json.dumps(selected, indent=2))
    print("\nTop 10:")
    print(eligible.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
