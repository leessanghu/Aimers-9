"""v13 feature set: all-history training versus recent-season windows.

The validation/test transform remains row-local and unchanged.  Only the rows used
to fit the predictive model are restricted, so this directly tests whether old
relationships hurt under the large league-level drift seen since 2019.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import inseason as INS_MOD
import features as FEATURES_MOD
from crosses import add_crosses
from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from platoon import build_platoon_table, transform_platoon

SEED = 42
INS_COLS = ["inseason_success_smooth", "inseason_ball_smooth",
            "inseason_reverse_smooth", "inseason_n", "inseason_is_first_appearance"]


def trackman_count_mix(main_df, seasons, k=30.0):
    """Prior-season (pitcher, count-state) pitch-choice deviations."""
    tm = pd.read_csv("../data/trackman_history.csv", usecols=[
        "season", "pitcher_trackman_id", "balls_before", "strikes_before",
        "pitch_type_group"])
    mapping = pd.read_csv("pitcher_map.csv").rename(columns={"tm_id": "pitcher_trackman_id"})
    tm = tm.merge(mapping[["pitcher_id", "pitcher_trackman_id"]],
                  on="pitcher_trackman_id", how="inner")
    groups = ["fastball", "breaking", "offspeed"]
    tm = tm[tm["pitch_type_group"].isin(groups)].copy()
    tm["_cs"] = tm["balls_before"] * 4 + tm["strikes_before"]
    tm["_one"] = 1.0
    for group in groups:
        tm[f"_{group}"] = (tm["pitch_type_group"] == group).astype(float)

    def cumulative(keys):
        g = tm.groupby(keys + ["season"])[["_one"] + [f"_{x}" for x in groups]].sum()
        return g.groupby(level=list(range(len(keys)))).cumsum().reset_index()

    cell = cumulative(["pitcher_id", "_cs"])
    glob = cumulative(["_cs"])
    idx_cell = pd.MultiIndex.from_arrays([
        main_df["pitcher_id"], main_df["balls_before"] * 4 + main_df["strikes_before"],
        main_df["season"] - 1])
    idx_glob = pd.MultiIndex.from_arrays([
        main_df["balls_before"] * 4 + main_df["strikes_before"], main_df["season"] - 1])

    def lookup(table, keys, col, idx):
        p = table.pivot_table(index=keys, columns="season", values=col, aggfunc="first")
        p = p.reindex(columns=seasons).ffill(axis=1)
        return p.stack(future_stack=True).reindex(idx).to_numpy(float)

    n = np.nan_to_num(lookup(cell, ["pitcher_id", "_cs"], "_one", idx_cell), nan=0.0)
    gn = np.nan_to_num(lookup(glob, ["_cs"], "_one", idx_glob), nan=0.0)
    out = pd.DataFrame(index=main_df.index)
    for group in groups:
        c = np.nan_to_num(lookup(cell, ["pitcher_id", "_cs"], f"_{group}", idx_cell), nan=0.0)
        gc = np.nan_to_num(lookup(glob, ["_cs"], f"_{group}", idx_glob), nan=0.0)
        prior = np.divide(gc, gn, out=np.full_like(gn, 1 / 3), where=gn > 0)
        out[f"tm_count_{group}_diff"] = (c + k * prior) / (n + k) - prior
    out["tm_count_mix_n"] = np.log1p(n)
    out["tm_count_mix_mapped"] = (n > 0).astype(float)
    return out


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    seasons = sorted(df["season"].unique().tolist())
    global_rate = float(df["control_success"].mean())

    season_end = build_season_end_table(df)
    piv = _pivots_from_table(season_end, seasons)
    lookup = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior = pd.Series(piv["rate"].reindex(lookup).to_numpy()).fillna(global_rate).to_numpy(float)

    INS_MOD.K_SMOOTH = 60.0
    x_ins = INS_MOD.transform_inseason(df, season_end, global_rate, seasons)
    x_plt = transform_platoon(df, build_platoon_table(df), prior, seasons, k=2500.0)
    x_inn = transform_inning(df, build_inning_table(df), build_inning_offset(df), prior,
                             seasons, k=570.0)
    x_ly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df),
                              seasons, k=30.0)
    x_tm = trackman_count_mix(df, seasons) if os.environ.get("TM_COUNT") == "1" else None

    base_smooth_k = float(os.environ.get("BASE_SMOOTH_K", "20"))
    FEATURES_MOD.SMOOTH_K_RATE = base_smooth_k
    if os.environ.get("GROUP_K") == "1":
        FEATURES_MOD.SMOOTH_K_BY_RATE = {
            "asof_pitcher_success_rate": 76.0,
            "asof_pitcher_reverse_rate": 51.0,
            "asof_pitcher_middle_rate": 158.0,
            "asof_pitcher_ball_rate": 290.0,
            "asof_pitcher_strike_rate": 386.0,
            "asof_batter_success_rate": 137.0,
            "asof_batter_middle_rate": 224.0,
            "asof_pitcher_fastball_rate": 24.0,
            "asof_pitcher_breaking_rate": 14.0,
            "asof_pitcher_offspeed_rate": 10.0,
        }
    fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
    tr_idx = df.index[df["season"] <= 2023]
    va_idx = df.index[df["season"] == 2024]

    def stack(base, idx):
        x = pd.concat([base.reset_index(drop=True),
                       x_ins.loc[idx, INS_COLS].reset_index(drop=True),
                       x_plt.loc[idx].reset_index(drop=True),
                       x_inn.loc[idx].reset_index(drop=True),
                       x_ly.loc[idx].reset_index(drop=True)] +
                      ([x_tm.loc[idx].reset_index(drop=True)] if x_tm is not None else []),
                      axis=1).astype(float)
        return pd.concat([x, add_crosses(x)], axis=1)

    xtr = stack(fold["X_train"], tr_idx)
    xva = stack(fold["X_valid"], va_idx)
    ytr = fold["y_train"]
    yva = fold["y_valid"]

    params = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                  l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                  n_iter_no_change=20, random_state=SEED)
    print(f"features={xtr.shape[1]} valid={len(xva):,} base_smooth_k={base_smooth_k:g} "
          f"group_k={bool(FEATURES_MOD.SMOOTH_K_BY_RATE)}", flush=True)
    first_seasons = [int(x) for x in os.environ.get("FIRST_SEASONS", "2019,2020,2021,2022").split(",")]
    for first_season in first_seasons:
        keep = (df.loc[tr_idx, "season"].to_numpy() >= first_season)
        started = time.time()
        model_kind = os.environ.get("MODEL", "hgb")
        xfit = xtr.loc[keep].reset_index(drop=True)
        yfit = ytr[keep]
        if model_kind == "cat":
            ti, ei = time_split_es(len(xfit))
            model = CatBoostClassifier(
                iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=15.0,
                random_seed=SEED, verbose=0, early_stopping_rounds=50,
                min_data_in_leaf=200, loss_function="Logloss")
            model.fit(xfit.iloc[ti], yfit[ti], eval_set=(xfit.iloc[ei], yfit[ei]))
        else:
            model = HistGradientBoostingClassifier(**params).fit(xfit, yfit)
        pred = model.predict_proba(xva)[:, 1]
        score = evaluate(yva, pred)["bss"] * 100000
        print(f"{model_kind} train {first_season}-2023 n={keep.sum():,} score={score:.1f} "
              f"pred_mean={pred.mean():.5f} sec={time.time()-started:.0f}", flush=True)
    print(f"total sec={time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
