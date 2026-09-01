"""피처 교차항 — 조립된 피처 행렬에서만 계산 (행 간 참조 전혀 없음).

설계 원칙: 트리는 곱/합/차를 스스로 근사할 수 있으므로 무작정 곱하면 노이즈만 는다.
트리가 '비효율적으로' 근사하는 것만 명시적으로 준다:
  (1) 비율 x/y  — 트리는 계단으로 근사해야 해서 분할을 많이 먹는다
  (2) 여러 항의 합 — 실력 + 좌우보정 + 이닝보정을 트리가 직접 더하려면 깊이가 필요
  (3) 대각선 상호작용 (실력 x 상황압박)

교차 대상은 CatBoost 중요도 상위 피처로 한정:
  kal_post 16.8 / platoon_diff 2.6 / asof_pitcher_success_rate_smooth 2.0 /
  same_hand 2.0 / reverse 2.0 / batter_success 1.9 / ball 1.6 / hand_matchup 1.3 /
  kal_post_minus_pred 1.2 / asof_pitcher_n 1.0 / prev5 1.0 / balls·strikes 0.9
"""

import numpy as np
import pandas as pd

EPS = 1e-6
CROSS_COLS = [
    "x_ability_here", "x_ability_x_count", "x_ability_x_pressure", "x_ability_x_inning",
    "x_kal_minus_career", "x_p_over_b", "x_ball_over_strike", "x_rev_over_succ",
    "x_mid_over_succ", "x_count_pressure", "x_platoon_x_samehand", "x_prev5_minus_career",
    "x_prev1_minus_prev5", "x_exp_x_ability",
]


def _g(X, name, default=0.0):
    return X[name].to_numpy(np.float64) if name in X.columns else np.full(len(X), default)


def add_crosses(X):
    """X(조립된 피처 DataFrame) -> 교차항 DataFrame. 학습/추론에서 동일하게 사용."""
    # 실력 추정: v10은 kal_post, v7c 계열은 inseason_success_smooth
    ability = _g(X, "kal_post") if "kal_post" in X.columns else _g(X, "inseason_success_smooth")
    plat = _g(X, "platoon_diff")
    inn_d = _g(X, "inning_diff")
    career = _g(X, "asof_pitcher_success_rate_smooth")
    batter = _g(X, "asof_batter_success_rate_smooth")
    ball = _g(X, "asof_pitcher_ball_rate_smooth")
    strike = _g(X, "asof_pitcher_strike_rate_smooth")
    rev = _g(X, "asof_pitcher_reverse_rate_smooth")
    mid = _g(X, "asof_pitcher_middle_rate_smooth")
    balls = _g(X, "balls_before")
    strikes = _g(X, "strikes_before")
    cnt = _g(X, "count_state")
    inning = _g(X, "inning")
    same = _g(X, "same_hand")
    n_exp = _g(X, "asof_pitcher_n")
    prev5 = _g(X, "asof_pitcher_prev5_game_success_rate")
    prev1 = _g(X, "asof_pitcher_prev1_game_success_rate")

    out = pd.DataFrame(index=X.index)
    # (2) 합: 이 상황에서의 기대 실력 = 실력 + 좌우보정 + 이닝보정
    here = ability + plat + inn_d
    out["x_ability_here"] = here
    # (3) 실력 x 상황
    pressure = balls - strikes
    out["x_count_pressure"] = pressure
    out["x_ability_x_count"] = here * cnt
    out["x_ability_x_pressure"] = here * pressure
    out["x_ability_x_inning"] = here * inning
    out["x_platoon_x_samehand"] = plat * same
    out["x_exp_x_ability"] = n_exp * here
    # (1) 비율
    out["x_p_over_b"] = career / (batter + EPS)
    out["x_ball_over_strike"] = ball / (strike + EPS)
    out["x_rev_over_succ"] = rev / (career + EPS)
    out["x_mid_over_succ"] = mid / (career + EPS)
    # 차이 (현재 대 과거)
    out["x_kal_minus_career"] = ability - career
    out["x_prev5_minus_career"] = prev5 - career
    out["x_prev1_minus_prev5"] = prev1 - prev5

    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float64)
