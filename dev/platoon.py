"""플래툰 스플릿 피처 — (투수, 타자손) 조건부 성공률.

왜 이게 '새 정보'인가:
  주최측 asof_* 컬럼은 전부 marginal(투수 전체 성공률)이다. 모델은 same_hand/hand_matchup으로
  '전역 평균 플래툰 효과'만 알 수 있고, "투수 A는 좌타에 유독 약하다"는 개인차는 볼 수 없다.
  노이즈 제거 후 이 개인차의 진짜 SD = 0.0438 (투수 실력 개인차 0.0555의 79%).
  -> 그 행의 컬럼만으로는 절대 계산 불가 = train 다른 행 집계가 필요한 진짜 정보.

leakage 안전성 (in-season과 동일 구조):
  각 행은 자기 투수의 '직전 시즌까지 누적' (pitcher, batter_hand) 집계만 조회한다.
  같은 시즌의 다른 행, test.csv의 다른 행은 전혀 참조하지 않는다.
  테이블은 fit(=train)에서만 만들고 transform은 조회만 한다.

shrinkage: 셀 성공률을 그 투수 자신의 marginal로 축소(경험적 베이즈).
  K = p(1-p)/Var(편차). 편차 SD = 0.0438/2 = 0.0219 -> K ≈ 0.2494/0.00048 ≈ 520.
"""

import numpy as np
import pandas as pd

PLATOON_COLS = ["platoon_diff", "platoon_n"]
K_PLATOON = 520.0


def build_platoon_table(df, target_col="control_success"):
    """(pitcher_id, batter_hand, season) -> 그 시즌 끝까지의 누적 (n, 성공수).

    train 라벨로 직접 집계한다. 시즌 축으로 누적하므로 '그 시즌 종료 시점' 값이 된다."""
    g = (df.groupby(["pitcher_id", "batter_hand", "season"])[target_col]
           .agg(s="sum", n="count").sort_index())
    cum = g.groupby(level=[0, 1]).cumsum()
    return cum.reset_index()


def _lookup(table, value_col, seasons_range, lookup_idx):
    """(pitcher, hand) x season 피벗 후 시즌축 ffill -> lookup_idx로 조회."""
    p = table.pivot_table(index=["pitcher_id", "batter_hand"], columns="season",
                          values=value_col, aggfunc="first")
    p = p.reindex(columns=seasons_range).ffill(axis=1)
    return p.stack(future_stack=True).reindex(lookup_idx).to_numpy()


def transform_platoon(df, platoon_table, pitcher_prior_rate, seasons_range, k=K_PLATOON):
    """df에 플래툰 파생 2개를 붙여 반환.

    pitcher_prior_rate: 각 행의 '직전 시즌 끝 시점' 투수 marginal 성공률 (in-season 모듈에서
    이미 계산한 값을 그대로 재사용). 셀 추정치를 이 값으로 축소한다."""
    lookup_idx = pd.MultiIndex.from_arrays(
        [df["pitcher_id"], df["batter_hand"], df["season"] - 1])

    s_cell = np.nan_to_num(_lookup(platoon_table, "s", seasons_range, lookup_idx).astype(np.float64), nan=0.0)
    n_cell = np.nan_to_num(_lookup(platoon_table, "n", seasons_range, lookup_idx).astype(np.float64), nan=0.0)

    prior = np.asarray(pitcher_prior_rate, dtype=np.float64)
    rate_smooth = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["platoon_diff"] = rate_smooth - prior   # 그 투수 자신 대비 이 손잡이에서의 편차 = 새 신호
    out["platoon_n"] = np.log1p(n_cell)
    return out


def export_stats(platoon_table, seasons_range, k=K_PLATOON):
    return {"platoon_table": platoon_table, "seasons_range": list(seasons_range), "k": float(k)}
