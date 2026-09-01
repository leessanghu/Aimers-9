"""Trackman 물리 프로필 피처 — (투수, 시즌) 누적 집계를 '직전 시즌'로 조회.

기존 Trackman 시도는 시즌별 평균을 그대로 붙였고(-14~-28), 최근엔 개별 지표의
예측 타당성만 쟀다(구속변화 r=-0.05, 릴리스산포 r=+0.06으로 전부 무의미).
하지만 그 판정 근거도 결국 로컬/상관 분석이었고, 로컬은 실제 부호를 못 맞춘 전력이 있다.
이번엔 물리 프로필 전체를 한 배치로 넣고 리더보드에서 직접 측정한다.

leakage 안전성: (pitcher, season) 누적을 만들고 각 행은 season-1까지만 조회한다.
Trackman은 2019~2024를 덮으므로 2025 test는 2024까지 누적을 본다. 행 간 참조 없음.

매핑 안 된 투수(약 24%)는 결측 -> 리그 평균으로 채우고 tm_missing 플래그를 준다.
"""

import numpy as np
import pandas as pd

TM_PATH = "../data/trackman_history.csv"
MAP_PATH = "pitcher_map.csv"

BASE_COLS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
             "extension", "rel_height", "rel_side"]
TM_COLS = (["tm_velo", "tm_velo_sd", "tm_spin", "tm_ivb", "tm_hb", "tm_ext", "tm_ext_sd",
            "tm_relh", "tm_rels", "tm_rel_sd", "tm_fb_share", "tm_n", "tm_velo_chg", "tm_missing"])


def _load_raw(tm_path=TM_PATH, map_path=MAP_PATH):
    m = pd.read_csv(map_path).sort_values("sim", ascending=False).drop_duplicates("tm_id")
    pmap = m.set_index("tm_id")["pitcher_id"]
    tm = pd.read_csv(tm_path, encoding="utf-8-sig",
                     usecols=["season", "pitcher_trackman_id", "pitch_type_group"] + BASE_COLS)
    tm["pitcher_id"] = tm["pitcher_trackman_id"].map(pmap)
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm["pitcher_id"].astype(int)
    return tm


def build_trackman_table(tm_path=TM_PATH, map_path=MAP_PATH):
    """(pitcher_id, season) -> 그 시즌까지의 누적 물리 프로필."""
    tm = _load_raw(tm_path, map_path)
    # 릴리스 산포는 구종 내 편차 기준 (구종마다 릴리스가 다르므로)
    for c, out in [("rel_height", "rh_c"), ("rel_side", "rs_c")]:
        tm[out] = tm[c] - tm.groupby(["pitcher_id", "season", "pitch_type_group"])[c].transform("mean")
    tm["rel_c2"] = tm["rh_c"] ** 2 + tm["rs_c"] ** 2
    tm["is_fb"] = (tm["pitch_type_group"].astype(str).str.upper().str.startswith("F")).astype(float)

    agg = {c: ["sum", "count"] for c in ["rel_speed", "spin_rate", "induced_vert_break",
                                         "horz_break", "extension", "rel_height", "rel_side",
                                         "rel_c2", "is_fb"]}
    g = tm.groupby(["pitcher_id", "season"]).agg(agg)
    g.columns = [f"{a}_{b}" for a, b in g.columns]
    # 분산용 제곱합
    for c in ["rel_speed", "extension"]:
        g[f"{c}_sq"] = tm.groupby(["pitcher_id", "season"])[c].apply(lambda s: float((s ** 2).sum()))

    cum = g.groupby(level=0).cumsum()
    n = cum["rel_speed_count"].replace(0, np.nan)

    out = pd.DataFrame(index=cum.index)
    out["tm_velo"] = cum["rel_speed_sum"] / n
    out["tm_spin"] = cum["spin_rate_sum"] / cum["spin_rate_count"].replace(0, np.nan)
    out["tm_ivb"] = cum["induced_vert_break_sum"] / cum["induced_vert_break_count"].replace(0, np.nan)
    out["tm_hb"] = cum["horz_break_sum"] / cum["horz_break_count"].replace(0, np.nan)
    out["tm_ext"] = cum["extension_sum"] / cum["extension_count"].replace(0, np.nan)
    out["tm_relh"] = cum["rel_height_sum"] / cum["rel_height_count"].replace(0, np.nan)
    out["tm_rels"] = cum["rel_side_sum"] / cum["rel_side_count"].replace(0, np.nan)
    out["tm_fb_share"] = cum["is_fb_sum"] / cum["is_fb_count"].replace(0, np.nan)
    out["tm_rel_sd"] = np.sqrt(np.clip(cum["rel_c2_sum"] / cum["rel_c2_count"].replace(0, np.nan), 0, None))
    out["tm_velo_sd"] = np.sqrt(np.clip(cum["rel_speed_sq"] / n - out["tm_velo"] ** 2, 0, None))
    ext_n = cum["extension_count"].replace(0, np.nan)
    out["tm_ext_sd"] = np.sqrt(np.clip(cum["extension_sq"] / ext_n - out["tm_ext"] ** 2, 0, None))
    out["tm_n"] = np.log1p(cum["rel_speed_count"])

    # 시즌별 평균 구속(누적 아님)으로 '직전 시즌 대비 변화' 계산
    season_velo = (g["rel_speed_sum"] / g["rel_speed_count"].replace(0, np.nan))
    out["tm_velo_chg"] = season_velo - season_velo.groupby(level=0).shift(1)
    return out.reset_index()


def transform_trackman(df, tm_table, fill_values, seasons_range):
    """각 행에 '직전 시즌까지' 누적 프로필을 붙인다."""
    cols = [c for c in TM_COLS if c != "tm_missing"]
    piv = {c: tm_table.pivot(index="pitcher_id", columns="season", values=c)
                      .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
           for c in cols}
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"].to_numpy(), df["season"].to_numpy() - 1])

    out = pd.DataFrame(index=df.index)
    miss = None
    for c in cols:
        v = pd.Series(piv[c].reindex(idx).to_numpy(), index=df.index)
        if miss is None:
            miss = v.isna()
        out[c] = v.fillna(fill_values[c]).astype(np.float64)
    out["tm_missing"] = miss.astype(np.float64)
    return out


def build_fill_values(tm_table):
    cols = [c for c in TM_COLS if c != "tm_missing"]
    return {c: float(tm_table[c].median(skipna=True)) if tm_table[c].notna().any() else 0.0 for c in cols}


def export_stats(tm_table, fill_values, seasons_range):
    return {"tm_table": tm_table, "fill_values": fill_values, "seasons_range": list(seasons_range)}
