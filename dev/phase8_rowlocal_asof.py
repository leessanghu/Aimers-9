"""Three-fold validation for current-row-only asof reconstruction features.

This intentionally avoids any valid/test row-to-row lookup. Every validation
row feature is computed from:
  1. that row's own asof_* values, and
  2. season-end tables fitted on the training fold only.

No sorting/rolling/shift over validation or test rows is used.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from inseason import build_season_end_table as build_v4_season_end
from inseason import transform_inseason
from inseason_v2 import build_global_rates, build_season_end_table, transform_inseason_v2
from metrics import evaluate

SEED = 42
DATA_PATH = "../data/train.csv"
FOLDS = [(2021, 2022), (2022, 2023), (2023, 2024)]
FOLD_WEIGHTS = {2022: 0.2, 2023: 0.3, 2024: 0.5}

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                  l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=SEED)
W_RF, W_HGB = 0.15, 0.85


def add_rowlocal_deltas(x, v2):
    out = pd.DataFrame(index=v2.index)
    ins = v2["inseason_success_smooth_k15"].to_numpy(np.float64)
    prev1 = x["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
    prev3 = x["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float64)
    prev5 = x["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
    out["prev1_success_minus_inseason"] = prev1 - ins
    out["prev3_success_minus_inseason"] = prev3 - ins
    out["prev5_success_minus_inseason"] = prev5 - ins
    out["prev1_success_minus_prev5"] = prev1 - prev5
    out["prev3_success_minus_prev5"] = prev3 - prev5
    return out


def fit_preds(x_train, y_train, x_valid):
    rf = RandomForestClassifier(**RF_PARAMS).fit(x_train, y_train)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(x_train, y_train)
    p_rf = rf.predict_proba(x_valid)[:, 1]
    p_hgb = hgb.predict_proba(x_valid)[:, 1]
    return {
        "rf": p_rf,
        "hgb": p_hgb,
        "rf015_hgb085": W_RF * p_rf + W_HGB * p_hgb,
    }


def add_scores(rows, valid_season, variant, y_valid, preds):
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
        print(f"    {variant:18s} {model:13s} BSS={m['bss']:.6f} "
              f"score={m['leaderboard_score']:.1f} pred_mean={np.mean(pred):.6f}", flush=True)


def weighted_summary(out):
    print("\n===== weighted summary (0.2/0.3/0.5) =====", flush=True)
    for model in ["rf", "hgb", "rf015_hgb085"]:
        base = None
        for variant in out["variant"].drop_duplicates():
            sub = out[(out["model"] == model) & (out["variant"] == variant)]
            wbss = sum(float(sub[sub["valid_season"] == s]["bss"].iloc[0]) * w
                       for s, w in FOLD_WEIGHTS.items())
            if variant == "v4_current":
                base = wbss
            delta = "" if base is None else f" delta={((wbss - base) * 100000):+.1f}"
            print(f"  {variant:18s} {model:13s} weighted_bss={wbss:.6f} "
                  f"score={wbss * 100000:.1f}{delta}", flush=True)


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
        x_train_base = fb.transform_train_oof(train_df).reset_index(drop=True)
        x_valid_base = fb.transform(valid_df).reset_index(drop=True)

        seasons_range = sorted(train_df["season"].unique().tolist())

        # Exact current deployed v4 feature block.
        v4_end = build_v4_season_end(train_df)
        global_success = float(train_df[TARGET_COL].mean())
        x_train_v4 = transform_inseason(train_df, v4_end, global_success, seasons_range).reset_index(drop=True)
        x_valid_v4 = transform_inseason(valid_df, v4_end, global_success, seasons_range).reset_index(drop=True)

        # Extended row-local reconstructions. These do not inspect valid/test peer rows.
        v2_end = build_season_end_table(train_df)
        global_rates = build_global_rates(train_df)
        x_train_v2 = transform_inseason_v2(train_df, v2_end, global_rates, seasons_range, k_smooth_list=(15,)).reset_index(drop=True)
        x_valid_v2 = transform_inseason_v2(valid_df, v2_end, global_rates, seasons_range, k_smooth_list=(15,)).reset_index(drop=True)

        delta_train = add_rowlocal_deltas(x_train_base, x_train_v2)
        delta_valid = add_rowlocal_deltas(x_valid_base, x_valid_v2)

        variants = {
            "v4_current": (
                pd.concat([x_train_base, x_train_v4], axis=1),
                pd.concat([x_valid_base, x_valid_v4], axis=1),
            ),
            "success_only": (
                pd.concat([x_train_base, x_train_v2[[
                    "inseason_success_smooth_k15", "inseason_n", "inseason_is_first_appearance"]]], axis=1),
                pd.concat([x_valid_base, x_valid_v2[[
                    "inseason_success_smooth_k15", "inseason_n", "inseason_is_first_appearance"]]], axis=1),
            ),
            "v4_plus_mid_strike": (
                pd.concat([x_train_base, x_train_v4, x_train_v2[[
                    "inseason_middle_smooth_k15", "inseason_strike_smooth_k15"]]], axis=1),
                pd.concat([x_valid_base, x_valid_v4, x_valid_v2[[
                    "inseason_middle_smooth_k15", "inseason_strike_smooth_k15"]]], axis=1),
            ),
            "v4_plus_deltas": (
                pd.concat([x_train_base, x_train_v4, delta_train], axis=1),
                pd.concat([x_valid_base, x_valid_v4, delta_valid], axis=1),
            ),
            "v4_plus_mid_strike_deltas": (
                pd.concat([x_train_base, x_train_v4, x_train_v2[[
                    "inseason_middle_smooth_k15", "inseason_strike_smooth_k15"]], delta_train], axis=1),
                pd.concat([x_valid_base, x_valid_v4, x_valid_v2[[
                    "inseason_middle_smooth_k15", "inseason_strike_smooth_k15"]], delta_valid], axis=1),
            ),
        }

        for name, (x_train, x_valid) in variants.items():
            print(f"  {name} features={x_train.shape[1]}", flush=True)
            add_scores(rows, valid_season, name, y_valid, fit_preds(x_train, y_train, x_valid))
        print(f"  elapsed={time.time() - t0:.0f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv("phase8_rowlocal_asof_3fold.csv", index=False, encoding="utf-8-sig")
    weighted_summary(out)
    print(f"\nsaved phase8_rowlocal_asof_3fold.csv total={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
