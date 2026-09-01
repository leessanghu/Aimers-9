"""Legal row-independent workload recovery from prev-game rate denominators."""
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


def infer_min_denominator(success_rate, middle_rate, max_q, chunk=4000):
    pairs = pd.DataFrame({"s": success_rate, "m": middle_rate})
    unique = pairs.dropna().drop_duplicates().reset_index(drop=True)
    q = np.arange(1, max_q + 1, dtype=np.float64)
    inferred = np.empty(len(unique), dtype=np.float64)
    for start in range(0, len(unique), chunk):
        z = unique.iloc[start:start + chunk]
        s = z["s"].to_numpy()[:, None]
        m = z["m"].to_numpy()[:, None]
        err = np.maximum(np.abs(s * q - np.rint(s * q)), np.abs(m * q - np.rint(m * q))) / q
        valid = err <= 5.1e-7
        inferred[start:start + len(z)] = np.where(valid.any(axis=1), valid.argmax(axis=1) + 1, err.argmin(axis=1) + 1)
    unique["den"] = inferred
    out = pairs.merge(unique, on=["s", "m"], how="left", sort=False)["den"]
    return out.to_numpy(dtype=np.float64)


def hidden_denominator_features(df):
    out = pd.DataFrame(index=df.index)
    for k, max_q in ((1, 160), (3, 480), (5, 800)):
        out[f"prev{k}_hidden_total_n"] = infer_min_denominator(
            df[f"asof_pitcher_prev{k}_game_success_rate"],
            df[f"asof_pitcher_prev{k}_game_middle_rate"], max_q,
        )
    out["prev3_hidden_avg_n"] = out["prev3_hidden_total_n"] / 3.0
    out["prev5_hidden_avg_n"] = out["prev5_hidden_total_n"] / 5.0
    out["prev1_vs_prev3_workload"] = out["prev1_hidden_total_n"] - out["prev3_hidden_avg_n"]
    out["prev3_vs_prev5_workload"] = out["prev3_hidden_avg_n"] - out["prev5_hidden_avg_n"]
    return out


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df[TARGET].mean())
    sr = sorted(df["season"].unique().tolist())
    hidden = hidden_denominator_features(df)
    print(f"hidden denominator features={hidden.shape[1]}", flush=True)
    print(hidden.describe().T[["count", "mean", "std", "min", "max"]], flush=True)

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
                          hidden.loc[i].reset_index(drop=True)], axis=1).astype(float)

    xtr, xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
    print(f"fit v18_plus_hidden_denominator7 X={xtr.shape}", flush=True)
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
    print(f"v18_plus_hidden_denominator7 score={max(0,raw):.2f} raw={raw:.2f} brier={m['brier_score']:.8f} pred_mean={p.mean():.6f}", flush=True)
    print(f"delta_vs_812.88={raw-812.88:+.2f}", flush=True)
    print(f"time={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
