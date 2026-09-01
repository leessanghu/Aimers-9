"""RESEARCH ONLY — DO NOT SUBMIT (uses relationships between test rows).

Test the hidden pitcher-game structure encoded by prev-game rate signatures.

The six prev1/3/5 x success/middle values are constant during a pitcher outing.
Their change points recover a causal outing id without using the target.
Despite being causal, this violates the competition's independent-test-row rule.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
TARGET = "control_success"
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
SIG = [f"asof_pitcher_prev{k}_game_{z}_rate" for k in (1, 3, 5) for z in ("success", "middle")]


def hidden_game_features(df):
    order = df.assign(_orig=np.arange(len(df))).sort_values(["pitcher_id", "row_num"]).reset_index(drop=True)
    same = order["pitcher_id"].eq(order["pitcher_id"].shift())
    for c in SIG:
        same &= order[c].eq(order[c].shift()) | (order[c].isna() & order[c].shift().isna())
    order["_game"] = (~same).cumsum()
    order["game_pitch_no"] = order.groupby("_game").cumcount().add(1).astype(float)

    games = order.groupby("_game", sort=False).agg(
        pitcher_id=("pitcher_id", "first"), game_n=("_orig", "size")
    )
    games["pitcher_game_no"] = games.groupby("pitcher_id").cumcount().add(1).astype(float)
    gp = games.groupby("pitcher_id")["game_n"]
    games["prev_game_n"] = gp.shift(1)
    games["prev3_game_n_mean"] = gp.transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    games["prev5_game_n_mean"] = gp.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())
    games["workload_vs_prev3"] = games["game_n"] * np.nan  # filled causally per row below

    order = order.join(games[["pitcher_game_no", "prev_game_n", "prev3_game_n_mean", "prev5_game_n_mean"]], on="_game")
    order["game_pitch_frac_prev3"] = order["game_pitch_no"] / order["prev3_game_n_mean"].clip(lower=1)
    cols = ["game_pitch_no", "pitcher_game_no", "prev_game_n", "prev3_game_n_mean",
            "prev5_game_n_mean", "game_pitch_frac_prev3"]
    out = order.set_index("_orig")[cols].sort_index()
    return out.astype(float), games


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df[TARGET].mean())
    sr = sorted(df["season"].unique().tolist())

    workload, games = hidden_game_features(df)
    print(f"hidden_games={len(games):,}", flush=True)
    print(workload.describe().T[["mean", "std", "min", "max"]], flush=True)

    se = build_season_end_table(df)
    dins = transform_inseason(df, se, g, sr)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
    dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
    dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
    dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
    dly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0)
    fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
    tr = df[df.season <= 2023].index
    va = df[df.season == 2024].index
    ti, ei = time_split_es(len(tr))
    ytr, yva = fold["y_train"], fold["y_valid"]

    def stack(i, bf):
        x = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                       dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                       dpt.loc[i].reset_index(drop=True)], axis=1).astype(float)
        return pd.concat([x, add_crosses(x), dly.loc[i].reset_index(drop=True),
                          workload.loc[i].reset_index(drop=True)], axis=1).astype(float)

    xtr, xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
    print(f"fit v18_plus_hidden_game6 X={xtr.shape}", flush=True)
    h = HistGradientBoostingClassifier(
        max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
        l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=20, random_state=SEED,
    ).fit(xtr, ytr)
    cb = CatBoostClassifier(
        iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
        random_seed=SEED, verbose=0, early_stopping_rounds=50,
        min_data_in_leaf=200, loss_function="Logloss",
    )
    cb.fit(xtr.iloc[ti], ytr[ti], eval_set=(xtr.iloc[ei], ytr[ei]))
    p = 0.5 * h.predict_proba(xva)[:, 1] + 0.5 * cb.predict_proba(xva)[:, 1]
    m = evaluate(yva, p)
    score = max(0, m["bss"] * 1e5)
    print(f"v18_plus_hidden_game6 score={score:.2f} brier={m['brier_score']:.8f} pred_mean={p.mean():.6f}", flush=True)
    print(f"time={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
