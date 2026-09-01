"""RESEARCH ONLY — DO NOT SUBMIT (uses relationships between test rows).

v18 + causal recent-pitch state recovered from cumulative integer counts.
The competition explicitly prohibits test-row rolling/expanding features.
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
EVENTS = ("success", "reverse", "middle", "ball", "strike")
WINDOWS = (5, 10, 20, 50)
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]


def recent_pitch_features(df):
    order = df.assign(_orig=np.arange(len(df))).sort_values(["pitcher_id", "row_num"]).reset_index(drop=True)
    same = order["pitcher_id"].eq(order["pitcher_id"].shift()) & order["asof_pitcher_n"].sub(order["asof_pitcher_n"].shift()).eq(1)
    out = pd.DataFrame(index=order.index)
    for event in EVENTS:
        total = np.rint(order[f"asof_pitcher_{event}_rate"].fillna(0).to_numpy() * order["asof_pitcher_n"].to_numpy())
        lag = pd.Series(total, index=order.index).sub(pd.Series(total, index=order.index).shift()).where(same)
        out[f"recent_{event}_lag1"] = lag
        grouped = lag.groupby(order["pitcher_id"], sort=False)
        for window in WINDOWS:
            out[f"recent_{event}_{window}"] = grouped.transform(
                lambda s, w=window: s.rolling(w, min_periods=1).mean()
            )
    # Contrasts remove stable pitcher ability and expose short-run movement.
    out["recent_success20_minus_career"] = out["recent_success_20"] - order["asof_pitcher_success_rate"]
    out["recent_ball20_minus_career"] = out["recent_ball_20"] - order["asof_pitcher_ball_rate"]
    out["recent_reverse20_minus_career"] = out["recent_reverse_20"] - order["asof_pitcher_reverse_rate"]
    out["_orig"] = order["_orig"]
    return out.set_index("_orig").sort_index().astype(float)


def main():
    t0 = time.time()
    valid_year = int(sys.argv[1]) if len(sys.argv) > 1 else 2024
    train_end = valid_year - 1
    compare_baseline = "--compare" in sys.argv
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df[TARGET].mean())
    sr = sorted(df["season"].unique().tolist())
    recent = recent_pitch_features(df)
    print(f"recent_features={recent.shape[1]}", flush=True)
    print(recent[["recent_success_lag1", "recent_success_5", "recent_success_20",
                  "recent_success20_minus_career"]].describe().T, flush=True)

    se = build_season_end_table(df)
    dins = transform_inseason(df, se, g, sr)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
    dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
    dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
    dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
    dly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0)
    fold = build_fold(df, train_end, valid_year, extra_features=None, seed=SEED, include_team_te=True)
    tr = df[df.season <= train_end].index
    va = df[df.season == valid_year].index
    ti, ei = time_split_es(len(tr))
    ytr, yva = fold["y_train"], fold["y_valid"]

    def stack(i, bf, include_recent=True):
        x = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                       dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                       dpt.loc[i].reset_index(drop=True)], axis=1).astype(float)
        parts = [x, add_crosses(x), dly.loc[i].reset_index(drop=True)]
        if include_recent:
            parts.append(recent.loc[i].reset_index(drop=True))
        return pd.concat(parts, axis=1).astype(float)

    def fit_score(xtr, xva, label):
        print(f"fit {label} X={xtr.shape}", flush=True)
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
        raw = m["bss"] * 1e5
        print(f"{label} raw_score={raw:.2f} clipped={max(0, raw):.2f} brier={m['brier_score']:.8f} pred_mean={p.mean():.6f}", flush=True)
        return raw

    xtr, xva = stack(tr, fold["X_train"], True), stack(va, fold["X_valid"], True)
    recent_score = fit_score(xtr, xva, f"v18_plus_recent_valid{valid_year}")
    if compare_baseline:
        xtr0, xva0 = stack(tr, fold["X_train"], False), stack(va, fold["X_valid"], False)
        baseline_score = fit_score(xtr0, xva0, f"v18_baseline_valid{valid_year}")
        print(f"delta_recent_minus_baseline={recent_score-baseline_score:+.2f}", flush=True)
    elif valid_year == 2024:
        print(f"delta_vs_812.88={recent_score-812.88:+.2f}", flush=True)
    print(f"time={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
