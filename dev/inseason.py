"""in-season(시즌 한정) 피처 — dev/phase5_inseason.py에서 검증된 로직을 재사용 가능한
모듈로 정리. FeatureBuilder와 별개로 두는 이유: FeatureBuilder는 K-fold OOF 구조를 쓰는데
이건 (pitcher, season) 단위 정적 테이블이라 성격이 다르다.

leakage 안전성: 각 행은 자기 투수의 '직전 시즌 끝 시점'만 참조 (같은 시즌의 다른 행은 전혀 안 씀).
season_end_table은 fit_df(=train)에서만 만들고, transform은 그 테이블을 조회만 한다.
"""

import numpy as np
import pandas as pd

K_SMOOTH = 15.0
GLOBAL_BALL_PRIOR = 0.4
GLOBAL_REVERSE_PRIOR = 0.25

INSEASON_COLS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
                 "inseason_n", "inseason_is_first_appearance"]


def build_season_end_table(df):
    """(pitcher_id, season) -> 그 시즌이 끝난 시점의 진짜 누적 n/성공/볼/역방향 횟수.

    asof_pitcher_n은 '이 투구 직전까지' 값이라 시즌 마지막 행 기준으론 그 마지막 투구
    자체가 누락된다(off-by-one). N_end/S_end는 그 마지막 투구의 실제 결과(control_success)를
    더해 정확히 보정한다. B_end/R_end는 그 투구가 볼성/역방향 중 무엇이었는지 행 단위로
    알 수 없어 보정하지 않는다(최대 투구 1개분 근사 오차, k=15 스무딩에서 영향 미미)."""
    if "row_num" not in df.columns:
        df = df.assign(row_num=df["row_id"].str.replace("TRAIN_", "", regex=False)
                       .str.replace("TEST_", "", regex=False).astype(int))
    sub = df.sort_values(["pitcher_id", "row_num"])
    last = sub.groupby(["pitcher_id", "season"], as_index=False).last()
    n_before_last = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    last_outcome = last["control_success"].to_numpy(np.float64)

    last["N_end"] = n_before_last + 1
    last["S_end"] = np.round(last["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64)
                             * n_before_last) + last_outcome
    last["B_end"] = np.round(last["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_before_last)
    last["R_end"] = np.round(last["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_before_last)
    last["prior_success_rate"] = last["S_end"] / last["N_end"].replace(0, np.nan)
    return last[["pitcher_id", "season", "N_end", "S_end", "B_end", "R_end", "prior_success_rate"]]


def _pivots_from_table(season_end_table, seasons_range):
    pivots = {}
    for col, val_col in [("N_end", "N_end"), ("S_end", "S_end"), ("B_end", "B_end"),
                         ("R_end", "R_end"), ("rate", "prior_success_rate")]:
        p = season_end_table.pivot(index="pitcher_id", columns="season", values=val_col)
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        pivots[col] = p.stack(future_stack=True)
    return pivots


def transform_inseason(df, season_end_table, global_success_rate, seasons_range):
    """df(train 또는 test)에 in-season 파생 피처 5개를 붙여서 반환.
    season_end_table/global_success_rate/seasons_range는 fit(=train)에서만 계산된 것을 그대로 사용."""
    out = df.copy()
    pivots = _pivots_from_table(season_end_table, seasons_range)

    lookup_idx = pd.MultiIndex.from_arrays([out["pitcher_id"], out["season"] - 1])
    for col in ["N_end", "S_end", "B_end", "R_end"]:
        vals = pivots[col].reindex(lookup_idx).to_numpy()
        out[f"_{col}"] = np.nan_to_num(vals.astype(np.float64), nan=0.0)
    prior_rate_vals = pivots["rate"].reindex(lookup_idx).to_numpy()
    prior_rate = pd.Series(prior_rate_vals).fillna(global_success_rate).to_numpy(np.float64)

    n_now = out["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    s_now = np.round(out["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now)
    b_now = np.round(out["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_now)
    r_now = np.round(out["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_now)

    n_season = np.clip(n_now - out["_N_end"].to_numpy(np.float64), 0, None)
    s_season = np.clip(s_now - out["_S_end"].to_numpy(np.float64), 0, None)
    b_season = np.clip(b_now - out["_B_end"].to_numpy(np.float64), 0, None)
    r_season = np.clip(r_now - out["_R_end"].to_numpy(np.float64), 0, None)

    rate_raw = np.divide(s_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
    ball_raw = np.divide(b_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
    rev_raw = np.divide(r_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)

    result = pd.DataFrame(index=df.index)
    result["inseason_success_smooth"] = (n_season * np.nan_to_num(rate_raw) + K_SMOOTH * prior_rate) / (n_season + K_SMOOTH)
    result["inseason_ball_smooth"] = (n_season * np.nan_to_num(ball_raw) + K_SMOOTH * GLOBAL_BALL_PRIOR) / (n_season + K_SMOOTH)
    result["inseason_reverse_smooth"] = (n_season * np.nan_to_num(rev_raw) + K_SMOOTH * GLOBAL_REVERSE_PRIOR) / (n_season + K_SMOOTH)
    result["inseason_n"] = np.log1p(n_season)
    result["inseason_is_first_appearance"] = (n_season == 0).astype(np.float64)
    return result


def export_stats(season_end_table, global_success_rate, seasons_range):
    return {"season_end_table": season_end_table, "global_success_rate": float(global_success_rate),
            "seasons_range": list(seasons_range)}
