"""in-season 피처의 빠진 라벨 차원 보충 — middle / strike + 커맨드 종합지수.

발견한 공백:
  inseason.py의 build_season_end_table은 N_end/S_end/B_end/R_end만 저장한다.
  즉 in-season 피처가 success/ball/reverse 3개 차원뿐이다.
  그런데 주최측 공식 컬럼에는 asof_pitcher_middle_rate, asof_pitcher_strike_rate도 있고,
  lastyear.py는 ly_middle을 이미 쓰고 있다. -> in-season만 middle/strike가 빠진 비대칭.

왜 middle이 강할 것으로 기대하는가:
  기존 SHAP magnitude 순위
    inseason_success_smooth  0.030  <- 1위
    inseason_reverse_smooth  0.022  <- 2위
    platoon_diff             0.004
    (최근 추가한 간접 대리지표들) 0.0004~0.0016
  상위 2개가 전부 '투수 자신의 직접 결과 x 당해시즌 x 큰 표본' 클래스다.
  middle('가운데 또는 위험 코스')는 reverse와 마찬가지로 제구 실패의 직접 지표이며
  같은 클래스에 속한다. 간접 대리지표(물리량/엔트로피/변동성)와는 성격이 다르다.

off-by-one: 원본과 동일하게 마지막 투구의 middle/strike 여부는 행 단위로 알 수 없어
보정하지 않는다(시즌 ~1000구 중 1구, k=15 스무딩에서 영향 미미).

leakage 안전성: inseason.py와 완전히 동일한 구조. 각 행은 자기 투수의 '직전 시즌 끝 시점'
누적만 참조하고 같은 시즌의 다른 행은 전혀 안 쓴다.
"""

import numpy as np
import pandas as pd

K_SMOOTH = 15.0

INSEASON_FULL_COLS = ["inseason_middle_smooth", "inseason_strike_smooth",
                      "inseason_cmd_index", "inseason_middle_minus_career"]


def build_season_end_table_full(df):
    """(pitcher_id, season) -> 시즌 종료 시점 누적 M_end(middle), K_end(strike)."""
    if "row_num" not in df.columns:
        df = df.assign(row_num=df["row_id"].str.replace("TRAIN_", "", regex=False)
                       .str.replace("TEST_", "", regex=False).astype(int))
    sub = df.sort_values(["pitcher_id", "row_num"])
    last = sub.groupby(["pitcher_id", "season"], as_index=False).last()
    n_before = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)

    last["M_end"] = np.round(last["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_before)
    last["K_end"] = np.round(last["asof_pitcher_strike_rate"].fillna(0).to_numpy(np.float64) * n_before)
    return last[["pitcher_id", "season", "M_end", "K_end"]]


def build_global_priors(df):
    return {
        "middle": float(df["asof_pitcher_middle_rate"].mean(skipna=True)),
        "strike": float(df["asof_pitcher_strike_rate"].mean(skipna=True)),
    }


def _pivot(table, col, seasons_range):
    p = table.pivot(index="pitcher_id", columns="season", values=col)
    return p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)


def transform_inseason_full(df, table_full, priors, seasons_range, n_end,
                            inseason_success=None, inseason_reverse=None, k=K_SMOOTH):
    """in-season middle/strike + 커맨드 종합지수.

    n_end: 각 행의 '직전 시즌 종료 시점 누적 투구수'. inseason.py의 pivots["N_end"]에서
           가져온 값을 그대로 넘긴다 (분모를 success 쪽과 정확히 일치시키기 위함).
    inseason_success/reverse: 기존 inseason 블록 결과를 재사용 (종합지수용).
    """
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    M_end = np.nan_to_num(_pivot(table_full, "M_end", seasons_range).reindex(idx)
                          .to_numpy().astype(np.float64), nan=0.0)
    K_end = np.nan_to_num(_pivot(table_full, "K_end", seasons_range).reindex(idx)
                          .to_numpy().astype(np.float64), nan=0.0)

    n_now = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    m_now = np.round(df["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_now)
    k_now = np.round(df["asof_pitcher_strike_rate"].fillna(0).to_numpy(np.float64) * n_now)

    n_season = np.clip(n_now - np.asarray(n_end, dtype=np.float64), 0, None)
    m_season = np.clip(m_now - M_end, 0, None)
    k_season = np.clip(k_now - K_end, 0, None)

    mid = (m_season + k * priors["middle"]) / (n_season + k)
    stk = (k_season + k * priors["strike"]) / (n_season + k)

    out = pd.DataFrame(index=df.index)
    out["inseason_middle_smooth"] = mid
    out["inseason_strike_smooth"] = stk

    # 커맨드 종합지수: 성공은 +, 의도반대/위험코스는 - 로 묶은 단일 축.
    # 트리는 '여러 항의 합'을 비효율적으로 근사하므로 명시적으로 준다 (crosses.py 철학).
    if inseason_success is not None and inseason_reverse is not None:
        out["inseason_cmd_index"] = (np.asarray(inseason_success, dtype=np.float64)
                                     - np.asarray(inseason_reverse, dtype=np.float64) - mid)
    else:
        out["inseason_cmd_index"] = np.nan

    # 커리어 middle 대비 당해시즌 편차 = middle 차원의 폼 신호
    career_mid = df["asof_pitcher_middle_rate"].fillna(priors["middle"]).to_numpy(np.float64)
    out["inseason_middle_minus_career"] = mid - career_mid

    return out[INSEASON_FULL_COLS].astype(np.float64)


def export_stats(table_full, priors, seasons_range, k=K_SMOOTH):
    return {"table_full": table_full, "priors": priors,
            "seasons_range": list(seasons_range), "k": float(k)}
