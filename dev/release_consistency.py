"""릴리스포인트/구속 일관성 — Trackman 원시 물리량의 투수별 '산포(SD)'를 새 축으로 사용.

지금까지 Trackman은 전부 '평균 제구력'(pitchtype.py: 구종별 성공률)만 뽑았다.
여긴 다른 축이다 — 폼이 매 투구 얼마나 일정한가(rel_height/rel_side SD, rel_speed SD).
가설: 릴리스가 더 일정한 투수가 제구도 더 좋다 (기계적 반복성).

leakage 안전성: trackman_history를 (pitcher, season) 누적으로만 집계하고, 각 행은
season-1까지 누적된 값만 조회한다 (lastyear.py와 동일 패턴). 매칭에 pitch-level 셀
유일성이 필요 없다 — 특정 투구의 구종을 알아낼 필요가 없고 그 투수의 그 시즌 전체
산포만 필요하기 때문에, pitchtype.py보다 커버리지가 훨씬 높다 (pitcher_map 매핑만 있으면 됨).
"""

import numpy as np
import pandas as pd

REL_COLS = ["rel_h_sd", "rel_s_sd", "rel_speed_sd", "rel_n"]
K_REL = 200.0


def build_release_table(tm_path="../data/trackman_history.csv", map_path="pitcher_map.csv"):
    m = pd.read_csv(map_path).sort_values("sim", ascending=False).drop_duplicates("tm_id")
    t2p = m.set_index("tm_id")["pitcher_id"]

    tm = pd.read_csv(tm_path, encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "rel_height", "rel_side", "rel_speed"])
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(t2p)
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm["pitcher_id"].astype(int)
    tm = tm.dropna(subset=["rel_height", "rel_side", "rel_speed"])
    tm["sq_h"] = tm["rel_height"] ** 2
    tm["sq_s"] = tm["rel_side"] ** 2
    tm["sq_v"] = tm["rel_speed"] ** 2

    g = tm.groupby(["pitcher_id", "season"]).agg(
        n=("rel_height", "count"),
        sum_h=("rel_height", "sum"), sumsq_h=("sq_h", "sum"),
        sum_s=("rel_side", "sum"), sumsq_s=("sq_s", "sum"),
        sum_v=("rel_speed", "sum"), sumsq_v=("sq_v", "sum"),
    ).sort_index()
    cum = g.groupby(level=0).cumsum().reset_index()
    return cum


def _sd_from_cum(cum_n, cum_sum, cum_sumsq):
    mean = np.divide(cum_sum, cum_n, out=np.zeros_like(cum_sum), where=cum_n > 0)
    var = np.divide(cum_sumsq, cum_n, out=np.zeros_like(cum_sum), where=cum_n > 0) - mean ** 2
    return np.sqrt(np.clip(var, 0, None))


def transform_release(df, rel_table, seasons_range, k=K_REL):
    cols = ["n", "sum_h", "sumsq_h", "sum_s", "sumsq_s", "sum_v", "sumsq_v"]
    pivots = {c: rel_table.pivot(index="pitcher_id", columns="season", values=c)
                          .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
              for c in cols}
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    vals = {c: np.nan_to_num(pivots[c].reindex(idx).to_numpy().astype(np.float64), nan=0.0) for c in cols}

    n = vals["n"]
    sd_h = _sd_from_cum(n, vals["sum_h"], vals["sumsq_h"])
    sd_s = _sd_from_cum(n, vals["sum_s"], vals["sumsq_s"])
    sd_v = _sd_from_cum(n, vals["sum_v"], vals["sumsq_v"])

    valid = n > 5
    gh, gs, gv = sd_h[valid].mean() if valid.any() else 0.0, sd_s[valid].mean() if valid.any() else 0.0, sd_v[valid].mean() if valid.any() else 0.0

    out = pd.DataFrame(index=df.index)
    out["rel_h_sd"] = (n * sd_h + k * gh) / (n + k)
    out["rel_s_sd"] = (n * sd_s + k * gs) / (n + k)
    out["rel_speed_sd"] = (n * sd_v + k * gv) / (n + k)
    out["rel_n"] = np.log1p(n)
    return out


def export_stats(rel_table, seasons_range, k=K_REL):
    return {"rel_table": rel_table, "seasons_range": list(seasons_range), "k": float(k)}
