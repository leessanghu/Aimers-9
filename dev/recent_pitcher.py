"""Prefix-safe recent pitcher state features reconstructed from asof counts.

The asof pitcher rates encode cumulative counts before each pitch:
    S_t = round(asof_pitcher_success_rate_t * asof_pitcher_n_t)

For a fixed window W, the last-W-pitch count is S_t - S_{t-W}, using only
earlier rows for the same pitcher in the same season. This mirrors what the
test asof stream reveals before each pitch and does not read target labels.
"""

import numpy as np
import pandas as pd


RECENT_WINDOWS = (50, 100, 200)
EPS = 1e-5


def _ensure_row_num(df):
    if "row_num" in df.columns:
        return df
    return df.assign(row_num=df["row_id"].str.replace("TRAIN_", "", regex=False)
                     .str.replace("TEST_", "", regex=False).astype(int))


def _logit(x):
    x = np.clip(x, EPS, 1.0 - EPS)
    return np.log(x / (1.0 - x))


def build_recent_success_features(df, inseason_success, windows=RECENT_WINDOWS, k_smooth=25.0):
    """Return recent success features for each row.

    Parameters
    ----------
    df:
        DataFrame containing row_id, pitcher_id, season, asof_pitcher_n and
        asof_pitcher_success_rate. Rows may include train/valid/test.
    inseason_success:
        Prefix-safe season-to-date success estimate for the same rows. This is
        used only as a shrinkage prior and for deviation features.
    windows:
        Pitch-count windows. If a pitcher has fewer than W same-season prior
        pitches, the available season-to-date prefix is used.
    k_smooth:
        Beta shrinkage strength toward inseason_success.
    """
    d = _ensure_row_num(df).copy()
    prior = np.asarray(inseason_success, dtype=np.float64)
    n_now = d["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    s_now = np.round(d["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now)

    d["_n_now"] = n_now
    d["_s_now"] = s_now
    sort_cols = ["pitcher_id", "season", "row_num"]
    ordered = d.sort_values(sort_cols)

    out_ordered = pd.DataFrame(index=ordered.index)
    ordered_prior = pd.Series(prior, index=d.index).loc[ordered.index].to_numpy(np.float64)

    grp = ordered.groupby(["pitcher_id", "season"], sort=False)
    for w in windows:
        n_prev = grp["_n_now"].shift(w)
        s_prev = grp["_s_now"].shift(w)
        n_start = grp["_n_now"].transform("first")
        s_start = grp["_s_now"].transform("first")

        n_prev = n_prev.fillna(n_start).to_numpy(np.float64)
        s_prev = s_prev.fillna(s_start).to_numpy(np.float64)

        n_recent = np.clip(ordered["_n_now"].to_numpy(np.float64) - n_prev, 0.0, float(w))
        s_recent = np.clip(ordered["_s_now"].to_numpy(np.float64) - s_prev, 0.0, n_recent)

        raw = np.divide(s_recent, n_recent, out=np.full_like(n_recent, np.nan), where=n_recent > 0)
        smooth = (s_recent + k_smooth * ordered_prior) / (n_recent + k_smooth)
        raw_filled = np.where(n_recent > 0, raw, ordered_prior)

        out_ordered[f"recent_success_{w}_smooth"] = smooth
        out_ordered[f"recent_success_{w}_minus_inseason"] = smooth - ordered_prior
        out_ordered[f"recent_success_{w}_logit_minus_inseason"] = _logit(smooth) - _logit(ordered_prior)
        out_ordered[f"recent_success_{w}_n"] = np.log1p(n_recent)
        out_ordered[f"recent_success_{w}_raw"] = raw_filled

    return out_ordered.loc[d.index].reset_index(drop=True)
