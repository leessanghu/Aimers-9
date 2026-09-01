"""in-season 피처 v2 — 확장판.

v1(5개, K=15 사전 스무딩) 대비 변경점:
  A) 사전 스무딩 대신 raw 성분 + n_season을 그대로 주고 트리가 shrinkage를 학습하게 함
  B) 시즌 간 추세(prior_season_rate - two_seasons_ago_rate) 추가
  C) middle_rate, strike_rate도 시즌 한정화 (기존엔 ball/reverse만 했음)
모든 계산은 그 투수의 '직전 시즌 끝 시점'(들)만 참조 — 같은 시즌 다른 행 절대 미사용.
"""

import numpy as np
import pandas as pd

RATE_COLS = ["success", "ball", "reverse", "middle", "strike"]
ASOF_RATE_MAP = {
    "success": "asof_pitcher_success_rate", "ball": "asof_pitcher_ball_rate",
    "reverse": "asof_pitcher_reverse_rate", "middle": "asof_pitcher_middle_rate",
    "strike": "asof_pitcher_strike_rate",
}
GLOBAL_PRIOR_FALLBACK = {"success": None, "ball": 0.4, "reverse": 0.25, "middle": 0.15, "strike": 0.4}


def build_season_end_table(df):
    """(pitcher_id, season) -> 시즌 끝 시점 진짜 누적치 (off-by-one 보정 포함, success만 정확 보정
    가능 — control_success 라벨로 그 마지막 투구의 실제 결과를 아니까. 나머지는 근사)."""
    if "row_num" not in df.columns:
        df = df.assign(row_num=df["row_id"].str.replace("TRAIN_", "", regex=False)
                       .str.replace("TEST_", "", regex=False).astype(int))
    sub = df.sort_values(["pitcher_id", "row_num"])
    last = sub.groupby(["pitcher_id", "season"], as_index=False).last()
    n_before = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    last_outcome = last["control_success"].to_numpy(np.float64)

    last["N_end"] = n_before + 1
    for key in RATE_COLS:
        col = ASOF_RATE_MAP[key]
        cnt = np.round(last[col].fillna(0).to_numpy(np.float64) * n_before)
        if key == "success":
            cnt = cnt + last_outcome  # 마지막 투구 실제 결과 정확 반영
        last[f"{key}_end"] = cnt
        last[f"{key}_prior_rate"] = last[f"{key}_end"] / last["N_end"].replace(0, np.nan)

    keep = ["pitcher_id", "season", "N_end"] + [f"{k}_end" for k in RATE_COLS] + [f"{k}_prior_rate" for k in RATE_COLS]
    return last[keep]


def _pivot_lookup(table, col, seasons_range, lookup_idx):
    p = table.pivot(index="pitcher_id", columns="season", values=col)
    p = p.reindex(columns=seasons_range).ffill(axis=1)
    stacked = p.stack(future_stack=True)
    return stacked.reindex(lookup_idx).to_numpy()


def transform_inseason_v2(df, season_end_table, global_rates, seasons_range, k_smooth_list=(15,)):
    """반환: raw 성분 피처들 + 지정된 K들에 대한 스무딩 버전 + 시즌 추세."""
    out = pd.DataFrame(index=df.index)
    season = df["season"].to_numpy()

    lookup_prev = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    lookup_prev2 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 2])

    ends_prev = {}
    for key in ["N_end"] + [f"{k}_end" for k in RATE_COLS]:
        vals = _pivot_lookup(season_end_table, key, seasons_range, lookup_prev)
        ends_prev[key] = np.nan_to_num(vals.astype(np.float64), nan=0.0)

    n_now = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    now_cnt = {}
    for key in RATE_COLS:
        col = ASOF_RATE_MAP[key]
        now_cnt[key] = np.round(df[col].fillna(0).to_numpy(np.float64) * n_now)

    n_season = np.clip(n_now - ends_prev["N_end"], 0, None)
    out["inseason_n"] = np.log1p(n_season)
    out["inseason_is_first_appearance"] = (n_season == 0).astype(np.float64)

    prior_rate = {}
    for key in RATE_COLS:
        cnt_season = np.clip(now_cnt[key] - ends_prev[f"{key}_end"], 0, None)
        raw = np.divide(cnt_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)
        gp = global_rates[key]
        pr_vals = _pivot_lookup(season_end_table, f"{key}_prior_rate", seasons_range, lookup_prev)
        pr = pd.Series(pr_vals).fillna(gp).to_numpy(np.float64)
        prior_rate[key] = pr

        # A) raw 성분 (콜드스타트는 prior로 채움, 모델이 n으로 신뢰도 판단)
        out[f"inseason_{key}_raw"] = np.nan_to_num(raw, nan=0.0)
        out[f"inseason_{key}_raw"] = np.where(n_season > 0, out[f"inseason_{key}_raw"], pr)

        # K별 스무딩 버전
        for k in k_smooth_list:
            out[f"inseason_{key}_smooth_k{k}"] = (n_season * np.nan_to_num(raw) + k * pr) / (n_season + k)

    # B) 시즌 간 추세: 직전시즌 성공률 - 전전시즌 성공률
    pr2_vals = _pivot_lookup(season_end_table, "success_prior_rate", seasons_range, lookup_prev2)
    pr2 = pd.Series(pr2_vals).fillna(global_rates["success"]).to_numpy(np.float64)
    out["season_trend_success"] = prior_rate["success"] - pr2
    out["prior_season_success_rate"] = prior_rate["success"]  # = 통산(직전시즌까지) 수준, 참고용 노출

    return out


def build_global_rates(df):
    return {key: float(df[ASOF_RATE_MAP[key]].mean(skipna=True)) for key in RATE_COLS}
