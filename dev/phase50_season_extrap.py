"""Season extrapolation audit on the v18 2023->2024 validation setup.

Compares the original unseen numeric season, inference-time capping at the last
training season, retraining without the raw season column, and target-mean
calibration using only past season-level outcomes.
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
INS = [
    "inseason_success_smooth",
    "inseason_ball_smooth",
    "inseason_reverse_smooth",
    "inseason_n",
    "inseason_is_first_appearance",
]


def score(y, p, label):
    m = evaluate(y, p)
    s = max(0, m["bss"] * 1e5)
    print(
        f"{label}: score={s:.2f} brier={m['brier_score']:.8f} "
        f"y_mean={np.mean(y):.6f} pred_mean={np.mean(p):.6f}",
        flush=True,
    )
    return s


def logit_shift_to_mean(p, target_mean):
    # Diagnostic helper for historical validation only. Never solve this shift
    # from the evaluation test predictions; deployment must use a fixed shift
    # learned entirely from train-era validation.
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    z = np.log(p / (1 - p))
    lo, hi = -5.0, 5.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if np.mean(1 / (1 + np.exp(-(z + mid)))) < target_mean:
            lo = mid
        else:
            hi = mid
    shift = (lo + hi) / 2
    return 1 / (1 + np.exp(-(z + shift))), shift


def forecast_mean(train_ref, target_season, window=None):
    annual = train_ref.groupby("season")[TARGET].mean().sort_index()
    if window is not None:
        annual = annual.iloc[-window:]
    x = annual.index.to_numpy(dtype=np.float64)
    y = annual.to_numpy(dtype=np.float64)
    slope, intercept = np.polyfit(x, y, 1)
    return float(intercept + slope * target_season), float(slope), annual


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

    def stack(i, bf):
        x = pd.concat(
            [
                bf.reset_index(drop=True),
                dins.loc[i, INS].reset_index(drop=True),
                dplt.loc[i].reset_index(drop=True),
                dinn.loc[i].reset_index(drop=True),
                dpt.loc[i].reset_index(drop=True),
            ],
            axis=1,
        ).astype(np.float64)
        return pd.concat([x, add_crosses(x), dly.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)

    def fit_models(xtr):
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
        return h, cb

    def predict(models, x):
        h, cb = models
        return 0.5 * h.predict_proba(x)[:, 1] + 0.5 * cb.predict_proba(x)[:, 1]

    xtr = stack(tr, fold["X_train"])
    xva = stack(va, fold["X_valid"])
    print(f"X={xtr.shape}; raw season columns={[c for c in xtr.columns if c == 'season']}", flush=True)

    print("\n[1] fit baseline", flush=True)
    base_models = fit_models(xtr)
    p_base = predict(base_models, xva)
    s_base = score(yva, p_base, "baseline_raw_unseen_season")

    print("\n[2] inference-time season cap", flush=True)
    xva_cap = xva.copy()
    if "season" in xva_cap.columns:
        xva_cap["season"] = float(xtr["season"].max())
    p_cap = predict(base_models, xva_cap)
    s_cap = score(yva, p_cap, "cap_2024_to_2023")

    print("\n[3] fit without raw season", flush=True)
    keep = [c for c in xtr.columns if c != "season"]
    noseason_models = fit_models(xtr[keep])
    p_noseason = predict(noseason_models, xva[keep])
    s_noseason = score(yva, p_noseason, "retrain_without_season")

    print("\n[4] past-only mean forecasts and calibration", flush=True)
    train_ref = df.loc[tr]
    for window in (None, 5, 4, 3):
        mu_hat, slope, annual = forecast_mean(train_ref, 2024, window=window)
        p_cal, shift = logit_shift_to_mean(p_base, mu_hat)
        label = "all" if window is None else f"last{window}"
        print(f"forecast_{label}: mu={mu_hat:.6f} slope={slope:+.6f} logit_shift={shift:+.6f}", flush=True)
        score(yva, p_cal, f"baseline_cal_linear_{label}")

    p_oracle, oracle_shift = logit_shift_to_mean(p_base, float(np.mean(yva)))
    print(f"oracle logit_shift={oracle_shift:+.6f} (diagnostic ceiling only)", flush=True)
    score(yva, p_oracle, "baseline_oracle_mean_ceiling")

    print("\nDELTAS", flush=True)
    print(f"cap={s_cap-s_base:+.2f} noseason={s_noseason-s_base:+.2f}", flush=True)
    print(f"time={time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
