"""타자 쪽 조건부 피처 — 투수에 대해 만든 in-season 차분 트릭을 타자에 그대로 적용.

배경: 투수 쪽은 inseason/lastyear/platoon/inning/count/pitchtype/volatility/form/trackman까지
전부 만들었는데, 타자 쪽은 주최측 공식 컬럼 4개(asof_batter_n/success_rate/middle_rate + diff)가
전부였다. phase65 오라클 천장: batter_id 단독 148 (pitcher 840 대비 작지만 미개척 영역).

phase70 스크리닝 (부분상관, 1시그마=0.39점):
  bat_inseason_smooth        +17.05  (6.6시그마)  <- 핵심
  bat_inseason_minus_career  +10.09  (5.1시그마)
  bat_ly_n                    +0.73  (1.4시그마)
  bat_ly_rate                 +0.55  (1.2시그마)
  bat_inseason_n              +0.05  (0.4시그마)
  블록 합동 +17.3

복원 방식: inseason.py와 동일한 누적 차분.
  n_season = asof_batter_n(현재행) - N_end(season-1)
  s_season = round(asof_batter_success_rate * asof_batter_n) - S_end(season-1)
  -> '이번 시즌에 이 타자가 상대한 투구들의 제구 성공률' = 순수 당해 시즌 신호

규칙 준수: 테이블은 train에서만 만들고 각 행은 자기 타자의 season-1 시점 누적만 조회한다.
현재 행의 asof_batter_* 는 주최측 공식 입력 피처다. test 행 간 참조 없음.
"""

import numpy as np
import pandas as pd

BATTER_COLS = ["bat_inseason_smooth", "bat_inseason_n", "bat_ly_rate", "bat_ly_n",
               "bat_inseason_minus_career"]
K_BATTER = 30.0


def build_batter_table(df, target_col="control_success"):
    """(batter_id, season) -> 그 시즌 끝까지의 누적 (S, N)."""
    g = (df.groupby(["batter_id", "season"])[target_col]
           .agg(S="sum", N="count").sort_index())
    return g.groupby(level=0).cumsum().reset_index()


def transform_batter(df, batter_table, seasons_range, global_rate, k=K_BATTER):
    pv = {}
    for c in ("S", "N"):
        p = batter_table.pivot(index="batter_id", columns="season", values=c)
        pv[c] = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)

    idx = pd.MultiIndex.from_arrays([df["batter_id"], df["season"] - 1])
    S_end = np.nan_to_num(pv["S"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
    N_end = np.nan_to_num(pv["N"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    n_now = df["asof_batter_n"].fillna(0).to_numpy(np.float64)
    s_now = np.round(df["asof_batter_success_rate"].fillna(0).to_numpy(np.float64) * n_now)
    n_seas = np.clip(n_now - N_end, 0, None)
    s_seas = np.clip(s_now - S_end, 0, None)

    career = df["asof_batter_success_rate"].fillna(global_rate).to_numpy(np.float64)

    out = pd.DataFrame(index=df.index)
    out["bat_inseason_smooth"] = (s_seas + k * global_rate) / (n_seas + k)
    out["bat_inseason_n"] = np.log1p(n_seas)
    out["bat_ly_rate"] = np.divide(S_end, np.maximum(N_end, 1.0),
                                   out=np.full(len(df), global_rate), where=N_end > 0)
    out["bat_ly_n"] = np.log1p(N_end)
    out["bat_inseason_minus_career"] = out["bat_inseason_smooth"].to_numpy() - career
    return out[BATTER_COLS].astype(np.float64)


def export_stats(batter_table, seasons_range, global_rate, k=K_BATTER):
    return {"batter_table": batter_table, "seasons_range": list(seasons_range),
            "global_rate": float(global_rate), "k": float(k)}
