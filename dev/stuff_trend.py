"""시즌 내 구위 추세 — 최근 한 달 평균 구속/스핀이 그 이전 시즌 누적 평균 대비 얼마나 변했는가.
피로/부상 신호 가설: 최근 구위가 떨어지는 투수는 제구도 흔들릴 수 있다.

leakage 안전성: train/test 행에는 정확한 날짜가 없고 game_month만 있다. 그래서 '그 행의
game_month보다 엄격히 이전 달(같은 시즌)'만 사용한다 — 같은 달 안의 다른 행(미래일 수 있음)은
전혀 참조하지 않는다. 즉:
  recent  = (month-2, month-1] 누적 - (month-2] 누적   (직전 1개월)
  early   = (0, month-2] 누적                           (그 이전 시즌 누적)
  trend = recent_mean - early_mean, recent 표본수로 축소(0쪽으로)
"""

import numpy as np
import pandas as pd

STUFF_TREND_COLS = ["trend_speed", "trend_spin", "trend_n_recent"]
K_TREND = 50.0
MONTHS = list(range(1, 13))


def build_stuff_month_table(tm_path="../data/trackman_history.csv", map_path="pitcher_map.csv"):
    m = pd.read_csv(map_path).sort_values("sim", ascending=False).drop_duplicates("tm_id")
    t2p = m.set_index("tm_id")["pitcher_id"]

    tm = pd.read_csv(tm_path, encoding="utf-8-sig",
                     usecols=["season", "game_month", "pitcher_trackman_id", "rel_speed", "spin_rate"])
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(t2p)
    tm = tm.dropna(subset=["pitcher_id", "rel_speed", "spin_rate"])
    tm["pitcher_id"] = tm["pitcher_id"].astype(int)

    g = tm.groupby(["pitcher_id", "season", "game_month"]).agg(
        n=("rel_speed", "count"),
        sum_speed=("rel_speed", "sum"),
        sum_spin=("spin_rate", "sum"),
    ).sort_index()
    cum = g.groupby(level=[0, 1]).cumsum().reset_index()
    return cum


def transform_stuff_trend(df, month_table, k=K_TREND):
    cols = ["n", "sum_speed", "sum_spin"]
    piv = {c: month_table.pivot_table(index=["pitcher_id", "season"], columns="game_month",
                                      values=c, aggfunc="first")
                          .reindex(columns=MONTHS).ffill(axis=1).stack(future_stack=True)
           for c in cols}

    pid = df["pitcher_id"].to_numpy()
    season = df["season"].to_numpy()
    month = df["game_month"].to_numpy()

    def lookup_at(m_offset):
        target_month = month - m_offset
        safe_month = np.clip(target_month, 1, 12)
        idx = pd.MultiIndex.from_arrays([pid, season, safe_month])
        out = {}
        for c in cols:
            vals = piv[c].reindex(idx).to_numpy().astype(np.float64)
            vals = np.nan_to_num(vals, nan=0.0)
            vals = np.where(target_month <= 0, 0.0, vals)
            out[c] = vals
        return out

    upto_prev = lookup_at(1)
    upto_prev2 = lookup_at(2)

    recent_n = np.clip(upto_prev["n"] - upto_prev2["n"], 0, None)
    recent_speed_sum = upto_prev["sum_speed"] - upto_prev2["sum_speed"]
    recent_spin_sum = upto_prev["sum_spin"] - upto_prev2["sum_spin"]

    early_n = upto_prev2["n"]
    early_speed_mean = np.divide(upto_prev2["sum_speed"], early_n, out=np.zeros_like(early_n), where=early_n > 0)
    early_spin_mean = np.divide(upto_prev2["sum_spin"], early_n, out=np.zeros_like(early_n), where=early_n > 0)

    recent_speed_mean = np.divide(recent_speed_sum, recent_n, out=np.zeros_like(recent_n), where=recent_n > 0)
    recent_spin_mean = np.divide(recent_spin_sum, recent_n, out=np.zeros_like(recent_n), where=recent_n > 0)

    valid = (recent_n > 0) & (early_n > 0)
    trend_speed_raw = np.where(valid, recent_speed_mean - early_speed_mean, 0.0)
    trend_spin_raw = np.where(valid, recent_spin_mean - early_spin_mean, 0.0)

    shrink = recent_n / (recent_n + k)
    out = pd.DataFrame(index=df.index)
    out["trend_speed"] = trend_speed_raw * shrink
    out["trend_spin"] = trend_spin_raw * shrink
    out["trend_n_recent"] = np.log1p(recent_n)
    return out


def export_stats(month_table, k=K_TREND):
    return {"month_table": month_table, "k": float(k)}
