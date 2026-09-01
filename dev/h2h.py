"""투수-타자 상대전적(H2H) 피처 — (pitcher_id, batter_id) 조건부 성공률.

구조는 platoon.py/inning_split.py와 동일한 패턴 (셀을 투수 자신의 marginal로 축소).
차이는 조건 축이 batter_hand/inning이 아니라 batter_id 그 자체라는 점 — 표본이
훨씬 희소하다 (과거 기록 있는 행 51.2%, 30구 이상은 18.5%).

leakage 안전성: 각 행은 자기 투수의 '직전 시즌까지 누적' (pitcher_id, batter_id) 집계만
조회한다. 같은 시즌의 다른 행, test.csv의 다른 행은 전혀 참조하지 않음.
"""

import numpy as np
import pandas as pd

H2H_COLS = ["h2h_dev", "h2h_n"]
K_H2H = 100.0


def build_h2h_table(df, target_col="control_success"):
    g = (df.groupby(["pitcher_id", "batter_id", "season"])[target_col]
           .agg(s="sum", n="count").sort_index())
    cum = g.groupby(level=[0, 1]).cumsum()
    return cum.reset_index()


def _lookup(table, value_col, seasons_range, lookup_idx):
    p = table.pivot_table(index=["pitcher_id", "batter_id"], columns="season",
                          values=value_col, aggfunc="first")
    p = p.reindex(columns=seasons_range).ffill(axis=1)
    return p.stack(future_stack=True).reindex(lookup_idx).to_numpy()


def transform_h2h(df, h2h_table, pitcher_prior_rate, seasons_range, k=K_H2H):
    lookup_idx = pd.MultiIndex.from_arrays(
        [df["pitcher_id"], df["batter_id"], df["season"] - 1])

    s_cell = np.nan_to_num(_lookup(h2h_table, "s", seasons_range, lookup_idx).astype(np.float64), nan=0.0)
    n_cell = np.nan_to_num(_lookup(h2h_table, "n", seasons_range, lookup_idx).astype(np.float64), nan=0.0)

    prior = np.asarray(pitcher_prior_rate, dtype=np.float64)
    rate_smooth = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["h2h_dev"] = rate_smooth - prior
    out["h2h_n"] = np.log1p(n_cell)
    return out


def export_stats(h2h_table, seasons_range, k=K_H2H):
    return {"h2h_table": h2h_table, "seasons_range": list(seasons_range), "k": float(k)}
