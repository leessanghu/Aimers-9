"""Time-opened 12 features used by v16.

These are row-safe snapshot differences built from pitcher season-end cumulative
tables and the current row's official asof columns.
"""

import numpy as np
import pandas as pd

TIME103_COLS = [
    "ly_success_rel", "cur_minus_ly_success", "ly_minus_old_success",
    "ly_reverse_rel", "cur_minus_ly_reverse", "ly_minus_old_reverse",
    "ly_ball_rel", "cur_minus_ly_ball", "ly_minus_old_ball",
    "ly_middle_rel", "cur_minus_ly_middle", "ly_minus_old_middle",
]


def league_season_rates(ly_table, seasons_range):
    rows = []
    for s in seasons_range:
        cur = ly_table[ly_table["season"] == s]
        if cur.empty:
            continue
        den = max(float(cur["N_end"].sum()), 1.0)
        rows.append({
            "season": s,
            "success": float(cur["S_end"].sum() / den),
            "reverse": float(cur["R_end"].sum() / den),
            "ball": float(cur["B_end"].sum() / den),
            "middle": float(cur["M_end"].sum() / den),
        })
    return pd.DataFrame(rows).set_index("season")


def _lookup_cum(df, pivots, season_offset):
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] + season_offset])
    out = {}
    for c in ["N_end", "S_end", "R_end", "B_end", "M_end"]:
        out[c] = np.nan_to_num(pivots[c].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
    return out


def _rates_from_counts(num, den, fallback):
    raw = np.divide(num, den, out=np.full_like(den, np.nan, dtype=np.float64), where=den > 0)
    return np.nan_to_num(raw, nan=fallback)


def transform_time103(df, ly_table, global_rates, seasons_range, k_cur=15.0, k_ly=30.0):
    pivots = {c: ly_table.pivot(index="pitcher_id", columns="season", values=c)
                          .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
              for c in ["N_end", "S_end", "R_end", "B_end", "M_end"]}
    league = league_season_rates(ly_table, seasons_range)

    c1 = _lookup_cum(df, pivots, -1)
    c2 = _lookup_cum(df, pivots, -2)

    n_now = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    now = {
        "N_end": n_now,
        "S_end": np.round(df["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now),
        "R_end": np.round(df["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_now),
        "B_end": np.round(df["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_now),
        "M_end": np.round(df["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_now),
    }

    n_cur = np.clip(now["N_end"] - c1["N_end"], 0, None)
    n_ly = np.clip(c1["N_end"] - c2["N_end"], 0, None)
    n_old = c2["N_end"]

    out = pd.DataFrame(index=df.index)
    prev_season = df["season"].to_numpy() - 1
    for key, col in [("success", "S_end"), ("reverse", "R_end"), ("ball", "B_end"), ("middle", "M_end")]:
        gm = float(global_rates[key])
        cnt_cur = np.clip(now[col] - c1[col], 0, None)
        cnt_ly = np.clip(c1[col] - c2[col], 0, None)
        cnt_old = np.clip(c2[col], 0, None)

        cur_raw = _rates_from_counts(cnt_cur, n_cur, gm)
        ly_raw = _rates_from_counts(cnt_ly, n_ly, gm)
        old_raw = _rates_from_counts(cnt_old, n_old, gm)

        cur_sm = (n_cur * cur_raw + k_cur * gm) / (n_cur + k_cur)
        ly_sm = (n_ly * ly_raw + k_ly * gm) / (n_ly + k_ly)
        old_sm = (n_old * old_raw + k_ly * gm) / (n_old + k_ly)

        rel = np.asarray([float(league.loc[s, key]) if s in league.index else gm
                          for s in prev_season], dtype=np.float64)
        out[f"ly_{key}_rel"] = ly_sm - rel
        out[f"cur_minus_ly_{key}"] = cur_sm - ly_sm
        out[f"ly_minus_old_{key}"] = ly_sm - old_sm

    return out[TIME103_COLS].astype(np.float64)


def export_stats(ly_table, global_rates, seasons_range, k_cur=15.0, k_ly=30.0):
    return {
        "ly_table": ly_table,
        "global_rates": global_rates,
        "seasons_range": list(seasons_range),
        "k_cur": float(k_cur),
        "k_ly": float(k_ly),
    }
