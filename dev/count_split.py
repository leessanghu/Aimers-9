"""카운트 조건부 스플릿 — (투수, count_state) 조건부 성공률. platoon.py/inning_split.py와 동일 구조.

왜 이게 새 정보인가 (phase65 오라클 천장 실측):
  pitcher_id 단독 천장(2024, 잡음보정) = 840
  pitcher x count_state 천장          = 1223   <- platoon(1134)/inning(1029)보다 높음
  즉 count_state 조건부는 이미 만들어놓은 platoon/inning보다도 잠재력이 큰데 아직 없었다.

  "3-0에서 그루빙하는 투수"와 "0-2에서 유인구를 못 던지는 투수"는 다른데, 지금 모델은
  count_state를 원시 피처로만 갖고 있어 '전역 카운트 효과'만 알고 개인차를 못 본다.

shrinkage: 투수 자신의 marginal 대비 편차의 노이즈보정 진짜 SD = 0.0168 (전체 train 실측).
  K = p(1-p)/Var(편차) = 0.2494/0.000283 ≈ 880.

leakage 안전성: platoon과 동일. 각 행은 자기 투수의 '직전 시즌까지 누적' (pitcher, count_state)
집계만 조회. 같은 시즌/test의 다른 행 참조 없음.
"""

import numpy as np
import pandas as pd

COUNT_COLS = ["count_diff", "count_n"]
K_COUNT = 880.0


def build_count_table(df, target_col="control_success"):
    d = df.copy()
    d["count_state"] = d["balls_before"] * 4 + d["strikes_before"]
    g = (d.groupby(["pitcher_id", "count_state", "season"])[target_col]
          .agg(s="sum", n="count").sort_index())
    cum = g.groupby(level=[0, 1]).cumsum()
    return cum.reset_index()


def _lookup(table, value_col, seasons_range, lookup_idx):
    p = table.pivot_table(index=["pitcher_id", "count_state"], columns="season",
                          values=value_col, aggfunc="first")
    p = p.reindex(columns=seasons_range).ffill(axis=1)
    return p.stack(future_stack=True).reindex(lookup_idx).to_numpy()


def transform_count(df, count_table, pitcher_prior_rate, seasons_range, k=K_COUNT):
    """df에 카운트 조건부 파생 2개를 붙여 반환.

    pitcher_prior_rate: 각 행의 '직전 시즌 끝 시점' 투수 marginal 성공률
    (in-season 모듈 재사용, platoon/inning과 동일 prior)."""
    cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
    lookup_idx = pd.MultiIndex.from_arrays([df["pitcher_id"], cs, df["season"] - 1])

    s_cell = np.nan_to_num(_lookup(count_table, "s", seasons_range, lookup_idx).astype(np.float64), nan=0.0)
    n_cell = np.nan_to_num(_lookup(count_table, "n", seasons_range, lookup_idx).astype(np.float64), nan=0.0)

    prior = np.asarray(pitcher_prior_rate, dtype=np.float64)
    rate_smooth = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["count_diff"] = rate_smooth - prior
    out["count_n"] = np.log1p(n_cell)
    return out


def export_stats(count_table, seasons_range, k=K_COUNT):
    return {"count_table": count_table, "seasons_range": list(seasons_range), "k": float(k)}
