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
INS = [
    "inseason_success_smooth",
    "inseason_ball_smooth",
    "inseason_reverse_smooth",
    "inseason_n",
    "inseason_is_first_appearance",
]


def transform_gametype_drift(df, train_ref, k=500.0):
    seasons = sorted(train_ref["season"].unique().tolist())
    mu = float(train_ref[TARGET].mean())

    season_global = train_ref.groupby("season")[TARGET].agg(s="sum", n="count")
    gt_season = train_ref.groupby(["game_type", "season"])[TARGET].agg(s="sum", n="count")
    gt_all = train_ref.groupby("game_type")[TARGET].agg(s="sum", n="count")

    season_rate = ((season_global["s"] + k * mu) / (season_global["n"] + k)).to_dict()
    gt_recent = ((gt_season["s"] + k * mu) / (gt_season["n"] + k)).to_dict()
    gt_career = ((gt_all["s"] + k * mu) / (gt_all["n"] + k)).to_dict()

    out = pd.DataFrame(index=df.index)
    prev = df["season"].to_numpy() - 1
    gt = df["game_type"].astype(str).to_numpy()

    recent = np.array([gt_recent.get((g, s), gt_career.get(g, mu)) for g, s in zip(gt, prev)], dtype=np.float64)
    seas = np.array([season_rate.get(s, mu) for s in prev], dtype=np.float64)
    career = np.array([gt_career.get(g, mu) for g in gt], dtype=np.float64)

    out["gt_recent_rate"] = recent
    out["gt_recent_offset"] = recent - seas
    out["gt_drift"] = recent - career
    return out


def main():
    t0 = time.time()
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df[TARGET].mean())
    sr = sorted(df["season"].unique().tolist())
    print(f"train={len(df):,} seasons={sr}", flush=True)

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

    def stack(i, bf, extra=None):
        X = pd.concat(
            [
                bf.reset_index(drop=True),
                dins.loc[i, INS].reset_index(drop=True),
                dplt.loc[i].reset_index(drop=True),
                dinn.loc[i].reset_index(drop=True),
                dpt.loc[i].reset_index(drop=True),
            ],
            axis=1,
        ).astype(np.float64)
        parts = [X, add_crosses(X), dly.loc[i].reset_index(drop=True)]
        if extra is not None:
            parts.append(extra.loc[i].reset_index(drop=True))
        return pd.concat(parts, axis=1).astype(np.float64)

    print("build gt drift features", flush=True)
    gt = transform_gametype_drift(df, df[df.season <= 2023], k=500.0)
    print(gt.loc[va].describe().T[["mean", "std", "min", "max"]], flush=True)

    def fit_score(Xtr, Xva, label):
        print(f"\nfit {label} X={Xtr.shape}", flush=True)
        h = HistGradientBoostingClassifier(
            max_depth=6,
            max_leaf_nodes=31,
            max_iter=500,
            learning_rate=0.03,
            l2_regularization=5.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=SEED,
        ).fit(Xtr, ytr)
        cb = CatBoostClassifier(
            iterations=3000,
            learning_rate=0.03,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=SEED,
            verbose=0,
            early_stopping_rounds=50,
            min_data_in_leaf=200,
            loss_function="Logloss",
        )
        cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
        p = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
        m = evaluate(yva, p)
        print(f"{label} score={max(0, m['bss'] * 1e5):.2f} brier={m['brier_score']:.8f} pred_mean={p.mean():.6f}", flush=True)
        return p, max(0, m["bss"] * 1e5)

    Xtr0 = stack(tr, fold["X_train"])
    Xva0 = stack(va, fold["X_valid"])
    p0, s0 = fit_score(Xtr0, Xva0, "v18_single_seed_baseline")

    Xtr1 = stack(tr, fold["X_train"], gt)
    Xva1 = stack(va, fold["X_valid"], gt)
    p1, s1 = fit_score(Xtr1, Xva1, "v18_plus_gt_drift3")

    print(f"\nDELTA {s1 - s0:+.2f}", flush=True)
    val = df.loc[va, ["game_type", TARGET]].copy()
    val["p0"] = p0
    val["p1"] = p1
    val["delta_p"] = p1 - p0
    print("\nby game_type", flush=True)
    print(
        val.groupby("game_type").agg(
            n=(TARGET, "size"),
            y=(TARGET, "mean"),
            p0=("p0", "mean"),
            p1=("p1", "mean"),
            dp=("delta_p", "mean"),
        ),
        flush=True,
    )
    print(f"\ntime={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
