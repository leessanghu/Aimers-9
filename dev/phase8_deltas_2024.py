"""Fast 2024 check for the only promising row-local variant: v4 + deltas."""

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
RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                  l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=SEED)
W_RF, W_HGB = 0.15, 0.85


def deltas(x_base, x_v2):
    ins = x_v2["inseason_success_smooth_k15"].to_numpy(np.float64)
    out = pd.DataFrame(index=x_v2.index)
    for k in [1, 3, 5]:
        out[f"prev{k}_success_minus_inseason"] = x_base[f"asof_pitcher_prev{k}_game_success_rate"].to_numpy(np.float64) - ins
    out["prev1_success_minus_prev5"] = (
        x_base["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
        - x_base["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
    )
    out["prev3_success_minus_prev5"] = (
        x_base["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float64)
        - x_base["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
    )
    return out


def fit_eval(x_train, y_train, x_valid, y_valid, tag):
    rf = RandomForestClassifier(**RF_PARAMS).fit(x_train, y_train)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(x_train, y_train)
    p_rf = rf.predict_proba(x_valid)[:, 1]
    p_hgb = hgb.predict_proba(x_valid)[:, 1]
    preds = {"rf": p_rf, "hgb": p_hgb, "rf015_hgb085": W_RF * p_rf + W_HGB * p_hgb}
    out = {}
    print(f"[{tag}]")
    for name, pred in preds.items():
        m = evaluate(y_valid, pred)
        out[name] = m["bss"]
        print(f"  {name:13s} BSS={m['bss']:.6f} score={m['leaderboard_score']:.1f} "
              f"pred_mean={np.mean(pred):.6f}", flush=True)
    return out


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    train_df = df[df["season"] <= 2023].reset_index(drop=True)
    valid_df = df[df["season"] == 2024].reset_index(drop=True)
    y_train = train_df[TARGET_COL].to_numpy()
    y_valid = valid_df[TARGET_COL].to_numpy()

    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(train_df)
    x_train_base = fb.transform_train_oof(train_df).reset_index(drop=True)
    x_valid_base = fb.transform(valid_df).reset_index(drop=True)

    seasons_range = sorted(train_df["season"].unique().tolist())
    v4_end = build_v4_season_end(train_df)
    x_train_v4 = transform_inseason(train_df, v4_end, float(train_df[TARGET_COL].mean()), seasons_range).reset_index(drop=True)
    x_valid_v4 = transform_inseason(valid_df, v4_end, float(train_df[TARGET_COL].mean()), seasons_range).reset_index(drop=True)

    v2_end = build_season_end_table(train_df)
    rates = build_global_rates(train_df)
    x_train_v2 = transform_inseason_v2(train_df, v2_end, rates, seasons_range, k_smooth_list=(15,)).reset_index(drop=True)
    x_valid_v2 = transform_inseason_v2(valid_df, v2_end, rates, seasons_range, k_smooth_list=(15,)).reset_index(drop=True)

    x_train_current = pd.concat([x_train_base, x_train_v4], axis=1)
    x_valid_current = pd.concat([x_valid_base, x_valid_v4], axis=1)
    x_train_delta = pd.concat([x_train_current, deltas(x_train_base, x_train_v2)], axis=1)
    x_valid_delta = pd.concat([x_valid_current, deltas(x_valid_base, x_valid_v2)], axis=1)

    cur = fit_eval(x_train_current, y_train, x_valid_current, y_valid, "v4_current")
    dd = fit_eval(x_train_delta, y_train, x_valid_delta, y_valid, "v4_plus_deltas")
    print("\n===== delta =====")
    for k in cur:
        print(f"  {k:13s} delta_bss={dd[k]-cur[k]:+.6f} score={(dd[k]-cur[k])*100000:+.1f}")
    print(f"total={time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
