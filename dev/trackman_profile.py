"""Trackman 물리량 -> 투수 커맨드 프로파일 (신규 정보원).

지금까지 trackman_history.csv(179만행 x 30컬럼, 338MB)에서 우리가 쓴 건 pitch_type_group 하나뿐이고
(pitchtype.py -> pt_pred/pt_dev/pt_n 3개), 물리 컬럼 8개(rel_speed, spin_rate, induced_vert_break,
horz_break, extension, rel_height, rel_side, zone_speed)는 100% 미사용이었다.

설계 원칙 — '제구'와 물리적으로 직결되는 것만:
  1) 릴리스포인트 반복성: 델리버리를 못 반복하면 공이 의도한 곳으로 안 간다.
     단 구종마다 릴리스가 다른 건 정상이므로 반드시 (투수 x 구종) '내부' SD를 쓴다.
     구종 간 차이까지 섞으면 '레퍼토리가 넓다'는 것과 '제구가 나쁘다'를 구분 못 한다.
  2) 구속/무브먼트 일관성: 같은 구종인데 구속·무브가 흔들리면 메커닉이 불안정하다.
  3) 무브먼트 크기: 많이 휘는 공일수록 제구가 물리적으로 어렵다 (stuff <-> command 트레이드오프).
  4) 경기 내 피로: pitch_no 대비 구속 하락 기울기. 반드시 (등판) 내부에서 demean해서
     '등판 간 차이'가 아니라 '등판 내부 감쇠'만 잡는다.
  5) 압박 반응: 3볼 카운트에서 릴리스 산포가 평소보다 커지는가.

규칙 준수:
  - 프로파일은 (pitcher_id, season) 누적 테이블로 만들고 각 행은 season-1까지만 조회한다
    (platoon/inning/lastyear와 동일 구조).
  - trackman은 2019~2024만 존재하고 2025는 애초에 제공되지 않으므로 미래 정보 유입 불가.
  - 추론 시 trackman 원본 불필요 (테이블을 아티팩트에 저장).
  - test 행 간 참조 없음.

커버리지: pitcher_map.csv가 train 행의 98.2% (2024 투수의 96.8%)를 커버한다.
  (pitchtype.py의 61.5%는 '행 단위 구종 확정매칭' 커버리지라 여기와 무관 — 여기는 투수 단위 집계)
"""

import numpy as np
import pandas as pd

TM_PATH = "../data/trackman_history.csv"
MAP_PATH = "pitcher_map.csv"

PHYS = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break",
        "extension", "rel_height", "rel_side", "zone_speed"]

USECOLS = ["season", "trackman_game_id", "pitch_no", "balls_before", "strikes_before",
           "pitcher_trackman_id", "pitch_type_group"] + PHYS

# 소표본 축소 강도 (구종셀 투구수 기준). SD 추정은 n이 작으면 매우 불안정해서 반드시 축소한다.
K_SD = 150.0
K_PROFILE = 200.0

PROFILE_COLS = [
    "tm_n",
    "tm_release_sd",      # 릴리스 2D 산포 (구종내) — 커맨드의 핵심 지표
    "tm_rel_h_sd", "tm_rel_s_sd", "tm_ext_sd",
    "tm_speed_sd", "tm_ivb_sd", "tm_hb_sd",
    "tm_break_mag",       # 평균 무브먼트 크기 (클수록 제구 어려움)
    "tm_speed_mean", "tm_spin_mean", "tm_ext_mean",
    "tm_rel_h_mean", "tm_rel_s_mean",   # 암슬롯
    "tm_velo_decay",      # 등판 내부 구속 감쇠 기울기 (피로)
    "tm_press_rel_sd",    # 3볼 카운트 릴리스 산포 - 평소 산포 (압박 반응)
]


def _load_trackman(tm_path=TM_PATH, map_path=MAP_PATH):
    m = pd.read_csv(map_path).sort_values("sim", ascending=False).drop_duplicates("tm_id")
    t2p = m.set_index("tm_id")["pitcher_id"]

    tm = pd.read_csv(tm_path, encoding="utf-8-sig", usecols=USECOLS)
    tm = tm.rename(columns={"pitcher_trackman_id": "tm_id"})
    tm["pitcher_id"] = tm["tm_id"].map(t2p)
    tm = tm.dropna(subset=["pitcher_id"])
    tm["pitcher_id"] = tm["pitcher_id"].astype(np.int64)
    return tm


def _within_type_sd(tm, col):
    """(투수, 시즌, 구종) 셀 내부 SD를 셀 투구수로 가중평균 -> 구종 간 차이를 제거한 순수 반복성.

    셀 SD는 n이 작을수록 불안정하므로 전역 평균 SD 쪽으로 K_SD만큼 축소한 뒤 합친다."""
    g = tm.groupby(["pitcher_id", "season", "pitch_type_group"])[col]
    cell = g.agg(["count", "std"]).reset_index()
    cell = cell[cell["count"] >= 2]
    gsd = float(cell["std"].median())
    cell["sd_sh"] = (cell["count"] * cell["std"].fillna(gsd) + K_SD * gsd) / (cell["count"] + K_SD)
    # 셀 투구수 가중평균 (분산 공간에서 합치는 게 맞지만 SD 공간 가중평균이 실무적으로 더 안정적)
    cell["wsum"] = cell["sd_sh"] * cell["count"]
    agg = cell.groupby(["pitcher_id", "season"]).agg(wsum=("wsum", "sum"), n=("count", "sum"))
    return (agg["wsum"] / agg["n"]).rename(f"sd_{col}")


def _velo_decay(tm):
    """등판(trackman_game_id) 내부에서 pitch_no 대비 rel_speed 기울기.

    등판 내부에서 두 변수를 모두 demean한 뒤 기울기를 구하므로 '등판 간 컨디션 차이'는
    자동으로 소거되고 '한 경기 안에서 얼마나 구속이 떨어지는가'만 남는다."""
    d = tm[["pitcher_id", "season", "trackman_game_id", "pitch_no", "rel_speed"]].dropna()
    key = ["pitcher_id", "season", "trackman_game_id"]
    gm = d.groupby(key)
    d = d.assign(
        x=d["pitch_no"] - gm["pitch_no"].transform("mean"),
        y=d["rel_speed"] - gm["rel_speed"].transform("mean"),
    )
    d["xy"] = d["x"] * d["y"]
    d["xx"] = d["x"] * d["x"]
    agg = d.groupby(["pitcher_id", "season"]).agg(xy=("xy", "sum"), xx=("xx", "sum"))
    slope = agg["xy"] / agg["xx"].replace(0, np.nan)
    return slope.rename("tm_velo_decay")


def _pressure_release_sd(tm):
    """3볼 카운트에서의 릴리스 산포 - 전체 릴리스 산포. 양수면 압박에서 흔들린다는 뜻."""
    d = tm[["pitcher_id", "season", "balls_before", "rel_height", "rel_side"]].dropna()
    d["r2"] = np.sqrt(d["rel_height"] ** 2 + d["rel_side"] ** 2)
    allsd = d.groupby(["pitcher_id", "season"])["r2"].agg(["std", "count"])
    p = d[d["balls_before"] >= 3]
    psd = p.groupby(["pitcher_id", "season"])["r2"].agg(["std", "count"])
    j = allsd.join(psd, how="left", lsuffix="_all", rsuffix="_p")
    gsd = float(j["std_all"].median())
    # 3볼 표본은 적으므로 전체 산포 쪽으로 축소
    k = 80.0
    psd_sh = (j["count_p"].fillna(0) * j["std_p"].fillna(gsd) + k * j["std_all"].fillna(gsd)) / \
             (j["count_p"].fillna(0) + k)
    return (psd_sh - j["std_all"].fillna(gsd)).rename("tm_press_rel_sd")


def build_trackman_profile(tm_path=TM_PATH, map_path=MAP_PATH, verbose=True):
    """(pitcher_id, season) -> 물리 프로파일 테이블. season은 '그 시즌에 관측된' 값."""
    tm = _load_trackman(tm_path, map_path)
    if verbose:
        print(f"  trackman {len(tm):,}행 (매핑된 투수 {tm.pitcher_id.nunique()}명)", flush=True)

    tm["break_mag"] = np.sqrt(tm["induced_vert_break"] ** 2 + tm["horz_break"] ** 2)

    base = tm.groupby(["pitcher_id", "season"]).agg(
        tm_n=("rel_speed", "size"),
        tm_break_mag=("break_mag", "mean"),
        tm_speed_mean=("rel_speed", "mean"),
        tm_spin_mean=("spin_rate", "mean"),
        tm_ext_mean=("extension", "mean"),
        tm_rel_h_mean=("rel_height", "mean"),
        tm_rel_s_mean=("rel_side", "mean"),
    )

    sds = {}
    for col, out in [("rel_height", "tm_rel_h_sd"), ("rel_side", "tm_rel_s_sd"),
                     ("extension", "tm_ext_sd"), ("rel_speed", "tm_speed_sd"),
                     ("induced_vert_break", "tm_ivb_sd"), ("horz_break", "tm_hb_sd")]:
        sds[out] = _within_type_sd(tm, col).rename(out)
        if verbose:
            print(f"  {out} 계산 완료", flush=True)

    prof = base.join(list(sds.values()), how="left")
    prof["tm_release_sd"] = np.sqrt(prof["tm_rel_h_sd"] ** 2 + prof["tm_rel_s_sd"] ** 2)
    prof = prof.join(_velo_decay(tm), how="left")
    prof = prof.join(_pressure_release_sd(tm), how="left")
    if verbose:
        print(f"  프로파일 {len(prof):,}개 (투수x시즌)", flush=True)
    return prof[PROFILE_COLS].reset_index()


def _expanding(prof, seasons_range):
    """시즌 누적(expanding) 프로파일. 각 시즌 값은 '그 시즌까지'의 투구수 가중 누적."""
    rows = []
    for (pid), grp in prof.groupby("pitcher_id"):
        grp = grp.sort_values("season")
        n_cum = 0.0
        acc = {c: 0.0 for c in PROFILE_COLS if c != "tm_n"}
        for _, r in grp.iterrows():
            n = float(r["tm_n"]) if np.isfinite(r["tm_n"]) else 0.0
            for c in acc:
                v = r[c]
                if np.isfinite(v):
                    acc[c] += v * n
            n_cum += n
            out = {"pitcher_id": pid, "season": int(r["season"]), "tm_n": n_cum}
            for c in acc:
                out[c] = acc[c] / n_cum if n_cum > 0 else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def export_stats(prof, seasons_range, k=K_PROFILE, lown_threshold=None):
    return {"profile": prof, "seasons_range": list(seasons_range), "k": float(k),
            "lown_threshold": float(lown_threshold) if lown_threshold is not None else None}


def add_lown_interactions(X_tm, asof_pitcher_n, threshold):
    """저표본 상호작용 — phase70에서 확인된 '물리량은 저표본 투수에게만 유효' 패턴을 명시적으로 준다.

    phase70 층화 검정 (asof_pitcher_n 4분위별 trackman 블록 증분, 1시그마=1.6):
        Q1 저표본(n<=1100)      +34.9   <- 20시그마 이상
        Q2 (1100<n<=2942)       +47.5   <- 30시그마
        Q3 (2942<n<=5557)        -6.7
        Q4 고표본(n>5557)        +3.1
    물리량(릴리스 일관성 등)은 제구력의 '원인'인데 우리는 '결과'(성공률 이력)를 직접 관측한다.
    결과가 충분히 쌓인 투수에게는 원인 정보가 무용지물이고, 이력이 부족한 투수에게만 가치가 있다.

    상호작용 형태 비교 (선형 증분):
        원본만                      6.1
        원본 + 원본x1/log1p(n)      11.0
        원본 + 원본xexp(-n/2000)    11.8
        원본 + 원본x저표본지시자     15.7   <- 채택
    """
    n = np.asarray(asof_pitcher_n, dtype=np.float64)
    lown = (np.nan_to_num(n, nan=0.0) <= threshold).astype(np.float64)

    cols = [c for c in X_tm.columns if c != "tm_matched"]
    out = pd.DataFrame(index=X_tm.index)
    out["tm_lown_flag"] = lown
    for c in cols:
        out[f"{c}_x_lown"] = X_tm[c].to_numpy(np.float64) * lown
    return out.astype(np.float64)


def transform_trackman(df, prof, seasons_range, k=K_PROFILE):
    """각 행에 자기 투수의 season-1 시점 누적 프로파일을 붙인다 (pivot+ffill+stack, 타 모듈과 동일 구조)."""
    exp = _expanding(prof, seasons_range)

    out = pd.DataFrame(index=df.index)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])

    # 전역 평균 (fit 시점 상수) — 미매칭/콜드스타트 투수의 fallback
    glob = {c: float(exp[c].median()) for c in PROFILE_COLS if c != "tm_n"}

    piv_n = exp.pivot_table(index="pitcher_id", columns="season", values="tm_n", aggfunc="first")
    piv_n = piv_n.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    n_cell = np.nan_to_num(piv_n.reindex(idx).to_numpy().astype(np.float64), nan=0.0)
    out["tm_n"] = np.log1p(n_cell)
    out["tm_matched"] = (n_cell > 0).astype(np.float64)

    for c in PROFILE_COLS:
        if c == "tm_n":
            continue
        p = exp.pivot_table(index="pitcher_id", columns="season", values=c, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
        v = p.reindex(idx).to_numpy().astype(np.float64)
        gm = glob[c]
        v = np.where(np.isfinite(v), v, gm)
        # 관측 투구수로 전역값 쪽 축소 (표본 적은 투수는 전역 프로파일에 가깝게)
        out[c] = (n_cell * v + k * gm) / (n_cell + k)

    return out.astype(np.float64)
