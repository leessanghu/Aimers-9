from __future__ import annotations

import numpy as np
import pandas as pd


def _logit(value: np.ndarray) -> np.ndarray:
    value = np.clip(value, 1e-5, 1.0 - 1e-5)
    return np.log(value / (1.0 - value))


def add_advanced_features(df: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    eps = 1e-5

    def rate(name: str) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").fillna(global_mean).to_numpy(np.float64)

    def number(name: str, fill: float = 0.0) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").fillna(fill).to_numpy(np.float64)

    pitcher = np.clip(rate("asof_pitcher_success_rate"), eps, 1.0 - eps)
    batter = np.clip(rate("asof_batter_success_rate"), eps, 1.0 - eps)
    pn = number("asof_pitcher_n")
    bn = number("asof_batter_n")
    p_rel = pn / (pn + 250.0)
    b_rel = bn / (bn + 250.0)
    pitcher_eb = p_rel * pitcher + (1.0 - p_rel) * global_mean
    batter_eb = b_rel * batter + (1.0 - b_rel) * global_mean
    matchup_logit = _logit(pitcher_eb) + _logit(batter_eb) - _logit(np.array([global_mean]))[0]

    p1 = rate("asof_pitcher_prev1_game_success_rate")
    p3 = rate("asof_pitcher_prev3_game_success_rate")
    p5 = rate("asof_pitcher_prev5_game_success_rate")
    m1 = rate("asof_pitcher_prev1_game_middle_rate")
    m3 = rate("asof_pitcher_prev3_game_middle_rate")
    m5 = rate("asof_pitcher_prev5_game_middle_rate")
    middle = rate("asof_pitcher_middle_rate")
    reverse = rate("asof_pitcher_reverse_rate")
    ball = rate("asof_pitcher_ball_rate")
    strike = rate("asof_pitcher_strike_rate")
    balls = number("balls_before")
    strikes = number("strikes_before")
    outs = number("outs_before")
    runners = number("num_runners_on")
    li = number("li", 1.0)
    score_diff = number("score_diff_pitcher_team")
    two_strike = (strikes == 2).astype(np.float64)
    three_ball = (balls == 3).astype(np.float64)
    risp = ((number("runner_on_2b") > 0) | (number("runner_on_3b") > 0)).astype(np.float64)
    same_hand = (number("pitcher_hand") == number("batter_hand")).astype(np.float64)

    out = pd.DataFrame(index=df.index)
    out["adv_matchup_prob"] = 1.0 / (1.0 + np.exp(-matchup_logit))
    out["adv_matchup_vs_global"] = out["adv_matchup_prob"] - global_mean
    out["adv_pitcher_eb"] = pitcher_eb
    out["adv_batter_eb"] = batter_eb
    out["adv_strength_gap"] = pitcher_eb - batter_eb
    out["adv_strength_product"] = (pitcher_eb - global_mean) * (batter_eb - global_mean)
    out["adv_pitcher_reliability"] = p_rel
    out["adv_batter_reliability"] = b_rel
    out["adv_experience_hmean"] = 2.0 * np.log1p(pn) * np.log1p(bn) / (np.log1p(pn) + np.log1p(bn) + eps)
    out["adv_experience_imbalance"] = np.log1p(pn) - np.log1p(bn)
    out["adv_recent_success_weighted"] = 0.5 * p1 + 0.3 * p3 + 0.2 * p5
    out["adv_recent_success_slope"] = 0.5 * (p1 - p3) + 0.5 * (p3 - p5)
    out["adv_recent_success_vs_career"] = out["adv_recent_success_weighted"] - pitcher
    out["adv_recent_middle_weighted"] = 0.5 * m1 + 0.3 * m3 + 0.2 * m5
    out["adv_recent_middle_slope"] = 0.5 * (m1 - m3) + 0.5 * (m3 - m5)
    out["adv_recent_middle_vs_career"] = out["adv_recent_middle_weighted"] - middle
    out["adv_command_margin"] = strike - ball - middle - reverse
    out["adv_danger_rate"] = middle + reverse
    out["adv_count_balance"] = balls - strikes
    out["adv_count_depth"] = balls + strikes
    out["adv_two_strike_pitcher"] = two_strike * pitcher_eb
    out["adv_three_ball_pitcher"] = three_ball * pitcher_eb
    out["adv_count_matchup"] = (balls - strikes) * (pitcher_eb - batter_eb)
    out["adv_pressure"] = np.log1p(np.maximum(li, 0.0)) * (1.0 + risp + 0.5 * runners)
    out["adv_pressure_pitcher"] = out["adv_pressure"] * (pitcher_eb - global_mean)
    out["adv_pressure_batter"] = out["adv_pressure"] * (batter_eb - global_mean)
    out["adv_baseout_threat"] = runners + 1.5 * risp + 0.25 * outs
    out["adv_score_pressure"] = np.log1p(np.maximum(li, 0.0)) / (1.0 + np.abs(score_diff))
    out["adv_platoon_matchup"] = same_hand * (pitcher_eb - batter_eb)
    return out.astype(np.float32)
