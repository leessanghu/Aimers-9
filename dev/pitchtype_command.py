"""구종별 커맨드 프로파일 확장 (Domain.md Direction A) — pitchtype.py의 성공률 전용
주변화(marginalization) 메커니즘을 reverse/middle/ball/strike까지 4개 더 확장한다.

pt_dev(구종 매칭 기반 성공률 편차)는 이번 세션에서 실측으로 검증된 유일한 Trackman 성공
사례(+6.7)다. 이 스크립트는 새 정보원을 찾는 게 아니라 그 성공 메커니즘 자체를 재사용:

  pt_{L}_pred = Σ_t P(구종=t | 투수,카운트) · P(라벨=L | 투수,구종=t)   (L ∈ reverse/middle/ball/strike)

라벨(reverse/middle/ball/strike)은 pitchlabels.py로 이미 100% 검증된 방식으로 복원한다
(asof_pitcher_n이 행마다 정확히 +1 증가하는 것을 이용한 차분 복원).

주의(v17 전례): 이 4개 라벨을 '일반 컨텍스트'로 조건부화한 버전(label-conditioned features)은
실측에서 -10.1/-3으로 실패했다. 이번 버전은 그것과 라벨 원천은 같지만 조건화 방식이
다르다 — pitchtype.py처럼 '구종 항등식 주변화'라는 검증된 구조를 그대로 쓰는 것이지,
새 컨텍스트 축을 만드는 게 아니다. 그래도 같은 원천 라벨을 쓰는 만큼 리스크는 완전히
0이 아니므로, 잔차가치 + 직접폴드 재확인 없이는 채택하지 않는다.

leakage 안전성: 라벨 복원과 테이블은 train에서만 계산, 각 행은 season-1까지만 조회.
"""

import numpy as np
import pandas as pd

from pitchlabels import LABELS, recover_pitch_labels
from pitchtype import TYPES, _TB, _KEY

PT_CMD_COLS = [f"pt_{name}_pred" for name in LABELS] + [f"pt_{name}_dev" for name in LABELS]


def build_matched_with_labels(train_df, tm_path="../data/trackman_history.csv", map_path="pitcher_map.csv"):
    """pitchtype.build_matched와 동일한 매칭이지만, reverse/middle/ball/strike 복원라벨도 같이 들고온다."""
    labels = recover_pitch_labels(train_df)

    m = pd.read_csv(map_path).sort_values("sim", ascending=False).drop_duplicates("tm_id")
    p2t = m.set_index("pitcher_id")["tm_id"]

    tr = train_df[["season", "game_month", "game_dayofweek", "inning", "top_bottom",
                   "balls_before", "strikes_before", "outs_before", "pitcher_id",
                   "control_success"]].copy()
    for name in LABELS:
        tr[f"lab_{name}"] = labels[f"lab_{name}"].to_numpy()
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
    keep = ["pitcher_id", "season", "count_state", "ptype", "control_success"] + [f"lab_{n}" for n in LABELS]
    return out[keep]


def build_command_tables(matched, seasons_range):
    """(투수,구종) 라벨별 발생률 / (투수,카운트) 구종믹스 / 전역 카운트믹스 — 전부 시즌 누적.
    mix/gmix는 pitchtype.py와 동일 정의라 재사용 가능(둘 다 매칭된 표본에서 계산)."""
    tables = {}
    for name in LABELS:
        sub = matched.dropna(subset=[f"lab_{name}"])
        ctrl = (sub.groupby(["pitcher_id", "ptype", "season"])[f"lab_{name}"]
                .agg(s="sum", n="count").sort_index())
        ctrl = ctrl.groupby(level=[0, 1]).cumsum().reset_index()
        gtype = (sub.groupby(["ptype", "season"])[f"lab_{name}"].agg(s="sum", n="count").sort_index())
        gtype = gtype.groupby(level=0).cumsum().reset_index()
        tables[name] = {"ctrl": ctrl, "gtype": gtype}

    mix = (matched.groupby(["pitcher_id", "count_state", "ptype", "season"]).size()
           .rename("n").sort_index())
    mix = mix.groupby(level=[0, 1, 2]).cumsum().reset_index()
    gmix = (matched.groupby(["count_state", "ptype", "season"]).size().rename("n").sort_index())
    gmix = gmix.groupby(level=[0, 1]).cumsum().reset_index()
    return {"labels": tables, "mix": mix, "gmix": gmix}


def _pv(tbl, index, value, seasons_range):
    p = tbl.pivot_table(index=index, columns="season", values=value, aggfunc="first")
    return p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)


_LABEL_ASOF_COL = {"reverse": "asof_pitcher_reverse_rate", "middle": "asof_pitcher_middle_rate",
                   "ball": "asof_pitcher_ball_rate", "strike": "asof_pitcher_strike_rate"}


def get_label_priors(df, global_label_rate):
    """각 행의 '이 투구 직전까지' 투수 marginal 라벨 발생률 (공식 asof_* 컬럼 그대로, causally safe)."""
    return {name: df[_LABEL_ASOF_COL[name]].fillna(global_label_rate[name]).to_numpy(np.float64)
            for name in LABELS}


def transform_command(df, tables, global_label_rate, seasons_range, k_control=340.0, k_mix=80.0):
    """구종을 주변화한 라벨별(reverse/middle/ball/strike) 기대치."""
    label_prior = get_label_priors(df, global_label_rate)
    pid = df["pitcher_id"].to_numpy()
    season = df["season"].to_numpy()
    cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
    n_rows = len(df)
    prev = season - 1

    gm_n = _pv(tables["mix"], ["count_state", "ptype"], "n", seasons_range)
    mx_n = _pv(tables["mix"], ["pitcher_id", "count_state", "ptype"], "n", seasons_range)

    out = pd.DataFrame(index=df.index)
    for name in LABELS:
        ctrl = tables["labels"][name]["ctrl"]
        gtype = tables["labels"][name]["gtype"]
        gt_s = _pv(gtype, "ptype", "s", seasons_range)
        gt_n = _pv(gtype, "ptype", "n", seasons_range)
        ct_s = _pv(ctrl, ["pitcher_id", "ptype"], "s", seasons_range)
        ct_n = _pv(ctrl, ["pitcher_id", "ptype"], "n", seasons_range)

        prior = np.asarray(label_prior[name], dtype=np.float64)
        num_pred = np.zeros(n_rows)
        den_mix = np.zeros(n_rows)

        for t in TYPES:
            gs = np.nan_to_num(gt_s.reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
            gn = np.nan_to_num(gt_n.reindex(pd.MultiIndex.from_arrays([[t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
            type_rate = np.divide(gs, gn, out=np.full(n_rows, global_label_rate[name]), where=gn > 0)

            cs_ = np.nan_to_num(ct_s.reindex(pd.MultiIndex.from_arrays([pid, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
            cn_ = np.nan_to_num(ct_n.reindex(pd.MultiIndex.from_arrays([pid, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
            anchor = np.clip(prior + (type_rate - global_label_rate[name]), 1e-6, 1 - 1e-6)
            ctrl_t = (cs_ + k_control * anchor) / (cn_ + k_control)

            pm = np.nan_to_num(mx_n.reindex(pd.MultiIndex.from_arrays([pid, cs, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
            gmx = np.nan_to_num(gm_n.reindex(pd.MultiIndex.from_arrays([cs, [t] * n_rows, prev])).to_numpy().astype(np.float64), nan=0.0)
            num_pred += (pm + k_mix * gmx) * ctrl_t
            den_mix += (pm + k_mix * gmx)

        pred = np.divide(num_pred, den_mix, out=prior.copy(), where=den_mix > 0)
        out[f"pt_{name}_pred"] = pred
        out[f"pt_{name}_dev"] = pred - prior

    return out


def export_stats(tables, global_label_rate, seasons_range, k_control=340.0, k_mix=80.0):
    return {"tables": tables, "global_label_rate": global_label_rate,
            "seasons_range": list(seasons_range), "k_control": float(k_control), "k_mix": float(k_mix)}
