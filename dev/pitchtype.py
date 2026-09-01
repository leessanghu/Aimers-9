"""투수x구종 제구력 — Trackman을 투구 단위로 매칭해 복원한 뒤 구종을 주변화(marginalize).

배경:
  주최측은 현재 투구의 구종을 주지 않는다(결과 누출 방지). 그런데 제구 성공률은 구종마다
  크게 다르다 — 매칭으로 확인한 전역값: fastball 0.541 / offspeed 0.513 / breaking 0.484.
  게다가 '어떤 구종을 던질지'는 카운트로 강하게 예측된다(0-2 패스트볼 44.6% vs 3-0 86.2%).

핵심 식 (항등식이므로 규칙 위반이 아니며, 현재 투구의 구종을 쓰지 않는다):
  P(성공|x) = Σ_t P(구종 t|x) · P(성공|구종 t, x)
  -> 두 항 모두 '과거 시즌 train'에서만 추정하고, 각 행은 조회만 한다.

측정 결과 (phase35/36):
  매칭 정밀도 99.5~99.7% (batter_hand는 매칭키에 안 쓴 독립 검증변수)
  커버리지 61.5% (전체 train 대비)
  투수x구종 제구력 진짜SD=0.0271 (platoon 0.0438과 inning 0.0209 사이) -> 상한 ~294점
  재현상관 r=+0.475 [0.433,0.514]  <- platoon 0.328보다 높음. 우리가 측정한 조건부 중 최고.

leakage 안전성: 모든 테이블은 (entity, season) 누적으로 만들고 각 행은 season-1까지만 조회한다.
  추론 시 Trackman 파일은 불필요(테이블이 아티팩트에 저장됨). test 행 간 참조 없음.
"""

import numpy as np
import pandas as pd

TARGET = "control_success"
TYPES = ["fastball", "breaking", "offspeed", "other"]
K_CONTROL = 340.0   # 투수x구종 제구력 축소 (진짜SD 0.0271 -> p(1-p)/var = 340)
K_MIX = 80.0        # 투수x카운트 구종믹스를 전역 카운트믹스로 축소
PT_COLS = ["pt_pred", "pt_dev", "pt_n"]

_TB = {"T": "Top", "B": "Bottom"}
_KEY = ["season", "game_month", "game_dayofweek", "tm_id", "inning", "_tb",
        "balls_before", "strikes_before", "outs_before"]


def build_matched(train_df, tm_path="../data/trackman_history.csv", map_path="pitcher_map.csv"):
    """train 행에 Trackman 구종을 붙인다. 셀 내 구종이 유일한 경우만 확정 매칭."""
    m = pd.read_csv(map_path).sort_values("sim", ascending=False).drop_duplicates("tm_id")
    p2t = m.set_index("pitcher_id")["tm_id"]

    tr = train_df[["season", "game_month", "game_dayofweek", "inning", "top_bottom",
                   "balls_before", "strikes_before", "outs_before", "pitcher_id", TARGET]].copy()
    tr["tm_id"] = tr["pitcher_id"].map(p2t)
    tr = tr.dropna(subset=["tm_id"])
    tr["tm_id"] = tr["tm_id"].astype(int)
    tr["_tb"] = tr["top_bottom"].astype(str).map(lambda v: _TB.get(v, v))

    tm = pd.read_csv(tm_path, encoding="utf-8-sig",
                     usecols=["season", "game_month", "game_dayofweek", "inning", "top_bottom",
                              "balls_before", "strikes_before", "outs_before",
                              "pitcher_trackman_id", "pitch_type_group"])
    tm = tm.rename(columns={"pitcher_trackman_id": "tm_id"})
    tm = tm[tm["tm_id"].isin(set(tr["tm_id"]))]
    tm["_tb"] = tm["top_bottom"].astype(str)

    agg = tm.groupby(_KEY).agg(n_type=("pitch_type_group", "nunique"),
                               ptype=("pitch_type_group", "first"))
    j = tr.join(agg, on=_KEY)
    out = j[j["n_type"] == 1].copy()
    out["count_state"] = out["balls_before"] * 4 + out["strikes_before"]
    out["ptype"] = out["ptype"].where(out["ptype"].isin(TYPES), "other")
    return out[["pitcher_id", "season", "count_state", "ptype", TARGET]]


def build_pitchtype_tables(matched, seasons_range):
    """(투수,구종) 제구력 / (투수,카운트) 구종믹스 / 전역 카운트믹스 — 전부 시즌 누적."""
    ctrl = (matched.groupby(["pitcher_id", "ptype", "season"])[TARGET]
            .agg(s="sum", n="count").sort_index())
    ctrl = ctrl.groupby(level=[0, 1]).cumsum().reset_index()

    mix = (matched.groupby(["pitcher_id", "count_state", "ptype", "season"]).size()
           .rename("n").sort_index())
    mix = mix.groupby(level=[0, 1, 2]).cumsum().reset_index()

    gmix = (matched.groupby(["count_state", "ptype", "season"]).size().rename("n").sort_index())
    gmix = gmix.groupby(level=[0, 1]).cumsum().reset_index()

    gtype = (matched.groupby(["ptype", "season"])[TARGET].agg(s="sum", n="count").sort_index())
    gtype = gtype.groupby(level=0).cumsum().reset_index()
    return {"ctrl": ctrl, "mix": mix, "gmix": gmix, "gtype": gtype}


def _pv(tbl, index, value, seasons_range):
    p = tbl.pivot_table(index=index, columns="season", values=value, aggfunc="first")
    return p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)


def transform_pitchtype(df, tables, prior_rate, global_rate, seasons_range,
                        k_control=K_CONTROL, k_mix=K_MIX):
    """구종을 주변화한 제구 기대치. 현재 투구의 구종은 사용하지 않는다."""
    pid = df["pitcher_id"].to_numpy()
    season = df["season"].to_numpy()
    cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
    prior = np.asarray(prior_rate, dtype=np.float64)
    n_rows = len(df)

    # 전역: 구종별 성공률 오프셋 (직전 시즌 누적)
    gt_s = _pv(tables["gtype"], "ptype", "s", seasons_range)
    gt_n = _pv(tables["gtype"], "ptype", "n", seasons_range)
    # 전역: (카운트,구종) 믹스
    gm_n = _pv(tables["gmix"], ["count_state", "ptype"], "n", seasons_range)
    # 투수별
    ct_s = _pv(tables["ctrl"], ["pitcher_id", "ptype"], "s", seasons_range)
    ct_n = _pv(tables["ctrl"], ["pitcher_id", "ptype"], "n", seasons_range)
    mx_n = _pv(tables["mix"], ["pitcher_id", "count_state", "ptype"], "n", seasons_range)

    prev = season - 1
    num_pred = np.zeros(n_rows)
    den_mix = np.zeros(n_rows)
    tot_n = np.zeros(n_rows)

    for t in TYPES:
        gs = np.nan_to_num(gt_s.reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        gn = np.nan_to_num(gt_n.reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        type_rate = np.divide(gs, gn, out=np.full(n_rows, global_rate), where=gn > 0)

        cs_ = np.nan_to_num(ct_s.reindex(pd.MultiIndex.from_arrays([pid, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        cn_ = np.nan_to_num(ct_n.reindex(pd.MultiIndex.from_arrays([pid, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        # 그 투수의 구종별 제구력: (투수 전체 실력 + 구종 전역 오프셋) 쪽으로 축소
        anchor = np.clip(prior + (type_rate - global_rate), 1e-6, 1 - 1e-6)
        ctrl_t = (cs_ + k_control * anchor) / (cn_ + k_control)

        pm = np.nan_to_num(mx_n.reindex(pd.MultiIndex.from_arrays([pid, cs, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        gmx = np.nan_to_num(gm_n.reindex(pd.MultiIndex.from_arrays([cs, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
        num_pred += (pm + k_mix * gmx) * ctrl_t     # 분자: 믹스가중 제구력 (정규화는 뒤에서)
        den_mix += (pm + k_mix * gmx)
        tot_n += cn_

    pred = np.divide(num_pred, den_mix, out=prior.copy(), where=den_mix > 0)
    out = pd.DataFrame(index=df.index)
    out["pt_pred"] = pred
    out["pt_dev"] = pred - prior          # 순수 신규 신호
    out["pt_n"] = np.log1p(tot_n)
    return out


def export_stats(tables, global_rate, seasons_range, k_control=K_CONTROL, k_mix=K_MIX):
    return {"tables": tables, "global_rate": float(global_rate),
            "seasons_range": list(seasons_range),
            "k_control": float(k_control), "k_mix": float(k_mix)}
