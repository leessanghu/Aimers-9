"""물리적 pitch-shape 군집 (구종그룹 내부) — 투수x군집 제구력.

기존 pitchtype.py(구종 3그룹: fastball/breaking/offspeed)의 확장. 같은 그룹 안에서
(구속, 회전수, 수직무브먼트, 수평무브먼트, 익스텐션, 릴리스높이, 릴리스좌우) 7개 물리량으로
군집화한 뒤, '투수x구종그룹 주효과'를 뺀 순수 투수x군집 상호작용만 남긴다.

leakage: 군집 중심(centroid)은 train 전체 물리량 분포로 학습(비지도, 라벨 무관이라 안전).
  제구력 테이블은 (투수, 구종그룹, 군집, season) 누적으로 train에서만 만들고,
  각 행은 직전 시즌까지만 조회한다. 현재 투구의 실제 군집은 쓰지 않고 구종그룹처럼 주변화한다.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from pitchtype import TYPES

PHYS_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
             "extension", "rel_height", "rel_side"]
N_CLUSTERS = 4  # 그룹당 세부 군집 수


def build_matched_with_phys(train_df, tm_path="../data/trackman_history.csv", map_path="pitcher_map.csv"):
    """pitchtype.build_matched와 동일 매칭 + 물리량 컬럼까지 붙여서 반환."""
    m = pd.read_csv(map_path).sort_values("sim", ascending=False).drop_duplicates("tm_id")
    p2t = m.set_index("pitcher_id")["tm_id"]

    tr = train_df[["season", "game_month", "game_dayofweek", "inning", "top_bottom",
                   "balls_before", "strikes_before", "outs_before", "pitcher_id", "control_success"]].copy()
    tr["tm_id"] = tr["pitcher_id"].map(p2t)
    tr = tr.dropna(subset=["tm_id"])
    tr["tm_id"] = tr["tm_id"].astype(int)
    tr["_tb"] = tr["top_bottom"].astype(str).map({"T": "Top", "B": "Bottom"}.get)

    tm = pd.read_csv(tm_path, encoding="utf-8-sig",
                     usecols=["season", "game_month", "game_dayofweek", "inning", "top_bottom",
                              "balls_before", "strikes_before", "outs_before",
                              "pitcher_trackman_id", "pitch_type_group"] + PHYS_COLS)
    tm = tm.rename(columns={"pitcher_trackman_id": "tm_id"})
    tm = tm[tm["tm_id"].isin(set(tr["tm_id"]))]
    tm["_tb"] = tm["top_bottom"].astype(str)

    key = ["season", "game_month", "game_dayofweek", "tm_id", "inning", "_tb",
           "balls_before", "strikes_before", "outs_before"]
    agg = tm.groupby(key).agg(n_type=("pitch_type_group", "nunique"),
                              ptype=("pitch_type_group", "first"),
                              **{c: (c, "mean") for c in PHYS_COLS})
    j = tr.join(agg, on=key)
    out = j[j["n_type"] == 1].copy()
    out["count_state"] = out["balls_before"] * 4 + out["strikes_before"]
    out["ptype"] = out["ptype"].where(out["ptype"].isin(TYPES), "other")
    return out.dropna(subset=PHYS_COLS)


def fit_shape_clusters(matched, n_clusters=N_CLUSTERS, seed=42):
    """구종그룹별로 물리량 표준화 + KMeans. train에서만 fit."""
    models = {}
    for t in TYPES:
        sub = matched[matched["ptype"] == t]
        if len(sub) < n_clusters * 50:
            continue
        sc = StandardScaler().fit(sub[PHYS_COLS])
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10).fit(sc.transform(sub[PHYS_COLS]))
        models[t] = (sc, km)
    return models


def assign_clusters(matched, models):
    lab = pd.Series(index=matched.index, dtype=object)
    for t, (sc, km) in models.items():
        m = matched["ptype"] == t
        if m.sum() == 0:
            continue
        z = sc.transform(matched.loc[m, PHYS_COLS])
        lab.loc[m] = [f"{t}_{c}" for c in km.predict(z)]
    return lab


def build_shape_tables(matched, shape_labels):
    """(pitcher, ptype+shape, season) 누적 제구력 + (ptype+shape, season) 전역 성공률."""
    d = matched[["pitcher_id", "ptype", "season", "control_success"]].copy()
    d["shape"] = shape_labels
    d = d.dropna(subset=["shape"])

    ctrl = (d.groupby(["pitcher_id", "shape", "season"])["control_success"]
            .agg(s="sum", n="count").reset_index())
    ctrl[["s", "n"]] = ctrl.groupby(["pitcher_id", "shape"])[["s", "n"]].cumsum()

    grp = (d.groupby(["ptype", "season"])["control_success"].agg(s="sum", n="count").reset_index())
    grp[["s", "n"]] = grp.groupby("ptype")[["s", "n"]].cumsum()

    gshape = (d.groupby(["shape", "season"])["control_success"].agg(s="sum", n="count").reset_index())
    gshape[["s", "n"]] = gshape.groupby("shape")[["s", "n"]].cumsum()

    pshape_freq = (d.groupby(["pitcher_id", "shape", "season"]).size().rename("n").reset_index())
    pshape_freq["n"] = pshape_freq.groupby(["pitcher_id", "shape"])["n"].cumsum()

    return {"ctrl": ctrl, "grp": grp, "gshape": gshape, "freq": pshape_freq}
