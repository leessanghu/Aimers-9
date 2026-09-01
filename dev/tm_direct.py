"""트랙맨 직접 피처 — 2군 활동 + 시즌간 물리 트렌드.

배경: 트랙맨에서 통계적으로 실재하는 신호 2종을 확인했으나, 현재 모델은 이를
프록시로만 간접 수신 중이었다.
    minor_share (2군 등판비율)  t=-5.04  <- 최강. 현재 tm_matched(Spearman 0.718)로 간접수신
    d_ext (익스텐션 시즌간 변화) t=+3.83  <- 현재 어떤 피처도 대응 없음
    spin/speed 수준             t=-3.10 / -2.34
프록시는 열화판이므로 깨끗한 값을 직접 준다. 또한 partial_gain은 선형 측정이라
트리의 비선형 활용 여지를 반영하지 못한다.

기존 tm_* 피처와의 차이:
  - 기존은 1군/2군 구분을 전혀 안 함(확인됨). 여기서 처음 분리.
  - 기존 tm_velo_decay는 '등판 내부' 감쇠. 여기 d_*는 '시즌 간' 트렌드로 완전히 다름.
  - train의 game_type=F 비율로는 대체 불가: train은 2군을 과소표집(13.2% vs 실제 24.4%).

leakage 안전: 전부 season-1 조회(직전 시즌까지의 트랙맨만 사용).
2025 테스트행은 2024 트랙맨을 조회 -> 트랙맨에 2025가 없어도 정상 동작.
lastyear.py / platoon.py 와 동일한 패턴. test 행 간 참조 없음 -> Rule §4 준수.
"""

import numpy as np
import pandas as pd

TM_DIRECT_COLS = [
    "tmd_minor_share", "tmd_minor_late", "tmd_minor_n", "tmd_major_n",
    "tmd_ext", "tmd_d_ext", "tmd_spin", "tmd_d_spin", "tmd_speed", "tmd_d_speed",
]
MINOR_PREFIX = ("MIN_", "KBO_", "ACE_")


def build_tm_direct_table(tm_path="../data/trackman_history.csv", map_path="pitcher_map.csv"):
    """(pitcher_id, season) -> 그 시즌의 2군비율/물리량 + 시즌간 변화."""
    pm = pd.read_csv(map_path).sort_values("sim", ascending=False).drop_duplicates("tm_id")
    t2p = {v: k for k, v in pm.set_index("pitcher_id")["tm_id"].items()}

    tm = pd.read_csv(tm_path, encoding="utf-8-sig",
                     usecols=["season", "game_month", "pitcher_trackman_id", "pitcher_team",
                              "rel_speed", "spin_rate", "extension"])
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(t2p)
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm["pitcher_id"].astype(int)
    tm["minor"] = tm["pitcher_team"].astype(str).str.startswith(MINOR_PREFIX).astype(np.float64)

    g = tm.groupby(["pitcher_id", "season"])
    t = g.agg(tmd_minor_share=("minor", "mean"),
              tmd_ext=("extension", "mean"),
              tmd_spin=("spin_rate", "mean"),
              tmd_speed=("rel_speed", "mean"),
              _n=("minor", "size"))
    t["tmd_minor_n"] = np.log1p(g["minor"].sum())
    t["tmd_major_n"] = np.log1p(t["_n"] - g["minor"].sum())

    late = (tm[tm["game_month"] >= 8].groupby(["pitcher_id", "season"])["minor"].mean()
            .rename("tmd_minor_late"))
    t = t.join(late)
    t["tmd_minor_late"] = t["tmd_minor_late"].fillna(t["tmd_minor_share"])

    t = t.reset_index().sort_values(["pitcher_id", "season"])
    pg = t.groupby("pitcher_id")
    for c in ["tmd_ext", "tmd_spin", "tmd_speed"]:
        t["tmd_d_" + c.split("_", 1)[1]] = t[c] - pg[c].shift(1)
    # 표본 적은 시즌은 신뢰 불가 -> 제외
    t = t[t["_n"] >= 100].drop(columns=["_n"])
    return t


def transform_tm_direct(df, table, seasons_range):
    """각 행에 그 투수의 season-1 트랙맨 요약을 붙인다."""
    piv = {}
    for c in TM_DIRECT_COLS:
        if c not in table.columns:
            continue
        p = table.pivot(index="pitcher_id", columns="season", values=c)
        piv[c] = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)

    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    out = pd.DataFrame(index=df.index)
    for c in TM_DIRECT_COLS:
        if c not in piv:
            out[c] = np.nan
            continue
        out[c] = pd.Series(piv[c].reindex(idx).to_numpy(), index=df.index)
    # 트랙맨 이력 없는 투수 구분용 플래그(결측을 정보로)
    out["tmd_missing"] = out["tmd_minor_share"].isna().astype(np.float64)
    return out.astype(np.float64)


TM_DIRECT_ALL = TM_DIRECT_COLS + ["tmd_missing"]


def export_stats(table, seasons_range):
    return {"table": table, "seasons_range": list(seasons_range)}
