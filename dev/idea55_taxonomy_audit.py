"""Recovered-label taxonomy audit for the next auxiliary-target axis.

Uses only official train columns.  The current-pitch reverse/middle/ball/strike
labels are recovered from the next row's as-of cumulative counts, exactly as in
the already deployed auxiliary-head training code.  Nothing here is used at
test-time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def recover(df: pd.DataFrame, col: str) -> np.ndarray:
    pid = df["pitcher_id"].to_numpy()
    n = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    order = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
    same_next = np.zeros(len(df), dtype=bool)
    same_next[order[:-1]] = pid[order][1:] == pid[order][:-1]
    cumulative = np.round(df[col].fillna(0).to_numpy(np.float64) * n)
    ordered = cumulative[order]
    delta = np.empty(len(df), dtype=np.float64)
    delta[:-1] = ordered[1:] - ordered[:-1]
    delta[-1] = np.nan
    delta[~same_next[order]] = np.nan
    out = np.empty(len(df), dtype=np.float64)
    out[order] = delta
    return out


df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
for short in ["reverse", "middle", "ball", "strike"]:
    df[short] = recover(df, f"asof_pitcher_{short}_rate")

valid = df[["reverse", "middle", "ball", "strike"]].notna().all(axis=1)
d = df.loc[valid, ["season", "control_success", "reverse", "middle", "ball", "strike"]].copy()
for col in ["control_success", "reverse", "middle", "ball", "strike"]:
    d[col] = d[col].round().astype(np.int8)
d["dangerous"] = ((d["reverse"] == 1) | (d["middle"] == 1)).astype(np.int8)
d["other"] = ((d["control_success"] + d["reverse"] + d["middle"]) == 0).astype(np.int8)
d["pattern"] = d[["control_success", "reverse", "middle", "ball", "strike"]].astype(str).agg("".join, axis=1)

print(f"valid={len(d):,}/{len(df):,} ({len(d)/len(df):.2%})")
print("pattern bits: success/reverse/middle/ball/strike")
pat = d.groupby("pattern").agg(n=("pattern", "size"), success=("control_success", "mean"))
pat["share"] = pat["n"] / len(d)
print(pat.sort_values("n", ascending=False).to_string(formatters={"share": "{:.3%}".format}))

print("\n2x2 dangerous x ball")
tab = d.groupby(["dangerous", "ball"]).agg(
    n=("control_success", "size"), success_rate=("control_success", "mean"),
    reverse_rate=("reverse", "mean"), middle_rate=("middle", "mean"),
)
tab["share"] = tab["n"] / len(d)
print(tab.to_string(formatters={"share": "{:.3%}".format}))

print("\n2x2 dangerous x strike")
tab = d.groupby(["dangerous", "strike"]).agg(
    n=("control_success", "size"), success_rate=("control_success", "mean"),
    ball_rate=("ball", "mean"),
)
tab["share"] = tab["n"] / len(d)
print(tab.to_string(formatters={"share": "{:.3%}".format}))

print("\nconditional target coverage/prevalence by season")
candidates = {
    "danger_success": np.where(d["dangerous"].eq(1), d["control_success"], np.nan),
    "danger_failure": np.where(d["dangerous"].eq(1), 1 - d["control_success"], np.nan),
    "ball_success": np.where(d["ball"].eq(1), d["control_success"], np.nan),
    "strike_failure": np.where(d["strike"].eq(1), 1 - d["control_success"], np.nan),
    "nondanger_notball": np.where(d["dangerous"].eq(0), 1 - d["ball"], np.nan),
    "danger_notball": np.where(d["dangerous"].eq(1), 1 - d["ball"], np.nan),
}
rows = []
for name, values in candidates.items():
    values = np.asarray(values, dtype=np.float64)
    for season in ["all", 2022, 2023, 2024]:
        mask = np.ones(len(d), dtype=bool) if season == "all" else d["season"].eq(season).to_numpy()
        z = values[mask]
        rows.append((name, season, np.isfinite(z).mean(), np.nanmean(z), np.nanstd(z)))
out = pd.DataFrame(rows, columns=["target", "season", "coverage", "mean", "sd"])
print(out.to_string(index=False, formatters={"coverage": "{:.2%}".format, "mean": "{:.4f}".format, "sd": "{:.4f}".format}))

