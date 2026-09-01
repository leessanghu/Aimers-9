"""타자 조건부 스플릿 — (타자, count_state) / (타자, 투수손) 조건부 + 타자 in-season middle.

배경 (phase65 오라클 천장, 잡음보정):
    pitcher x count        1223   -> count_split.py 로 보유
    pitcher x batter_hand  1134   -> platoon.py 로 보유
    pitcher x inning       1029   -> inning_split.py 로 보유
    batter x count          509   -> 없음  <- 여기
    batter_id               148   -> in-season만 부분 보유

투수 쪽은 platoon/inning/count/pitchtype/volatility/form/trackman까지 전부 만들었는데
타자 쪽은 batterform.py의 in-season 하나뿐이었다. 그런데 v27에서 bat_inseason_smooth가
+17.1(6.6시그마)로 최근 추가분 중 최대였다 -> 타자 쪽이 남은 최대 미개척지.

구조는 platoon.py/count_split.py와 완전히 동일하다:
  각 행은 자기 '타자'의 직전 시즌까지 누적 (batter, ctx) 셀을 조회하고,
  그 타자 자신의 marginal 성공률로 축소한 뒤 편차를 낸다.
  -> 전역 카운트 효과나 전역 플래툰 효과가 아니라 '이 타자만의 조건부 개인차'가 남는다.

leakage 안전성: 테이블은 train에서만 만들고 season-1 조회만 한다. test 행 간 참조 없음.
"""

import numpy as np
import pandas as pd

BCOUNT_COLS = ["bcount_diff", "bcount_n"]
BPLATOON_COLS = ["bplatoon_diff", "bplatoon_n"]

# K는 phase74에서 노이즈보정 편차 SD를 실측해 산출 (K = p(1-p)/Var(편차))
#   batter x count       : cells=9,213  median_n=35  진짜SD=0.01397 -> K=1278
#   batter x pitcher_hand: cells=1,599  median_n=200 진짜SD=0.01002 -> K=2486
K_BCOUNT = 1278.0
K_BPLATOON = 2486.0


def build_batter_marginal(df, target_col="control_success"):
    """(batter_id, season) -> 시즌 종료 시점 누적 성공률. 조건부 축소의 기준점."""
    g = (df.groupby(["batter_id", "season"])[target_col]
           .agg(s="sum", n="count").sort_index())
    cum = g.groupby(level=0).cumsum().reset_index()
    cum["rate"] = cum["s"] / cum["n"].replace(0, np.nan)
    return cum


def lookup_batter_prior(df, marginal, seasons_range, global_rate):
    p = marginal.pivot(index="batter_id", columns="season", values="rate")
    p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    idx = pd.MultiIndex.from_arrays([df["batter_id"], df["season"] - 1])
    v = p.reindex(idx).to_numpy().astype(np.float64)
    return pd.Series(v).fillna(global_rate).to_numpy(np.float64)


# ----------------------------------------------------------------------
# (타자, count_state)
# ----------------------------------------------------------------------

def build_bcount_table(df, target_col="control_success"):
    d = df.copy()
    d["count_state"] = d["balls_before"] * 4 + d["strikes_before"]
    g = (d.groupby(["batter_id", "count_state", "season"])[target_col]
          .agg(s="sum", n="count").sort_index())
    return g.groupby(level=[0, 1]).cumsum().reset_index()


def transform_bcount(df, table, batter_prior, seasons_range, k=K_BCOUNT):
    cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
    idx = pd.MultiIndex.from_arrays([df["batter_id"], cs, df["season"] - 1])

    def lk(col):
        p = table.pivot_table(index=["batter_id", "count_state"], columns="season",
                              values=col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        return np.nan_to_num(p.stack(future_stack=True).reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    s_cell, n_cell = lk("s"), lk("n")
    prior = np.asarray(batter_prior, dtype=np.float64)
    rate = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["bcount_diff"] = rate - prior
    out["bcount_n"] = np.log1p(n_cell)
    return out.astype(np.float64)


# ----------------------------------------------------------------------
# (타자, 투수손) — 타자 입장의 플래툰
# ----------------------------------------------------------------------

def build_bplatoon_table(df, target_col="control_success"):
    g = (df.groupby(["batter_id", "pitcher_hand", "season"])[target_col]
           .agg(s="sum", n="count").sort_index())
    return g.groupby(level=[0, 1]).cumsum().reset_index()


def transform_bplatoon(df, table, batter_prior, seasons_range, k=K_BPLATOON):
    idx = pd.MultiIndex.from_arrays([df["batter_id"], df["pitcher_hand"], df["season"] - 1])

    def lk(col):
        p = table.pivot_table(index=["batter_id", "pitcher_hand"], columns="season",
                              values=col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        return np.nan_to_num(p.stack(future_stack=True).reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    s_cell, n_cell = lk("s"), lk("n")
    prior = np.asarray(batter_prior, dtype=np.float64)
    rate = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["bplatoon_diff"] = rate - prior
    out["bplatoon_n"] = np.log1p(n_cell)
    return out.astype(np.float64)


# ----------------------------------------------------------------------
# 타자 in-season middle (batterform.py의 success 버전과 같은 차분 트릭)
# ----------------------------------------------------------------------

BAT_MID_COLS = ["bat_inseason_middle", "bat_middle_minus_career"]


def build_batter_middle_table(df):
    """(batter_id, season) -> 시즌 종료 시점 누적 middle 카운트."""
    if "row_num" not in df.columns:
        df = df.assign(row_num=df["row_id"].str.replace("TRAIN_", "", regex=False)
                       .str.replace("TEST_", "", regex=False).astype(int))
    sub = df.sort_values(["batter_id", "row_num"])
    last = sub.groupby(["batter_id", "season"], as_index=False).last()
    n_before = last["asof_batter_n"].fillna(0).to_numpy(np.float64)
    last["BM_end"] = np.round(last["asof_batter_middle_rate"].fillna(0).to_numpy(np.float64) * n_before)
    last["BN_end"] = n_before
    return last[["batter_id", "season", "BM_end", "BN_end"]]


def transform_batter_middle(df, table, seasons_range, global_middle, k=30.0):
    idx = pd.MultiIndex.from_arrays([df["batter_id"], df["season"] - 1])

    def lk(col):
        p = table.pivot(index="batter_id", columns="season", values=col)
        p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
        return np.nan_to_num(p.reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    BM_end, BN_end = lk("BM_end"), lk("BN_end")
    n_now = df["asof_batter_n"].fillna(0).to_numpy(np.float64)
    m_now = np.round(df["asof_batter_middle_rate"].fillna(0).to_numpy(np.float64) * n_now)
    n_seas = np.clip(n_now - BN_end, 0, None)
    m_seas = np.clip(m_now - BM_end, 0, None)

    mid = (m_seas + k * global_middle) / (n_seas + k)
    career = df["asof_batter_middle_rate"].fillna(global_middle).to_numpy(np.float64)

    out = pd.DataFrame(index=df.index)
    out["bat_inseason_middle"] = mid
    out["bat_middle_minus_career"] = mid - career
    return out.astype(np.float64)


def export_stats(bcount_table, bplatoon_table, bmid_table, marginal, seasons_range,
                 global_rate, global_middle, k_bcount=K_BCOUNT, k_bplatoon=K_BPLATOON, k_bmid=30.0):
    return {"bcount_table": bcount_table, "bplatoon_table": bplatoon_table,
            "bmid_table": bmid_table, "marginal": marginal,
            "seasons_range": list(seasons_range), "global_rate": float(global_rate),
            "global_middle": float(global_middle), "k_bcount": float(k_bcount),
            "k_bplatoon": float(k_bplatoon), "k_bmid": float(k_bmid)}
