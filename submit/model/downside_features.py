from __future__ import annotations

import numpy as np
import pandas as pd


def add_downside_features(df: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    eps = 1e-6

    def num(name: str, fill: float = 0.0) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").fillna(fill).to_numpy(np.float64)

    def rate(name: str) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").fillna(global_mean).to_numpy(np.float64)

    balls = num("balls_before")
    strikes = num("strikes_before")
    outs = num("outs_before")
    inning = num("inning", 5.0)
    month = num("game_month", 6.0)
    runners = num("num_runners_on")
    li = np.maximum(num("li", 1.0), 0.0)
    score_diff = num("score_diff_pitcher_team")
    r1 = num("runner_on_1b")
    r2 = num("runner_on_2b")
    r3 = num("runner_on_3b")
    risp = ((r2 > 0) | (r3 > 0)).astype(np.float64)
    loaded = ((r1 > 0) & (r2 > 0) & (r3 > 0)).astype(np.float64)

    pitcher = rate("asof_pitcher_success_rate")
    p1 = rate("asof_pitcher_prev1_game_success_rate")
    p3 = rate("asof_pitcher_prev3_game_success_rate")
    p5 = rate("asof_pitcher_prev5_game_success_rate")
    middle = rate("asof_pitcher_middle_rate")
    reverse = rate("asof_pitcher_reverse_rate")
    ball = rate("asof_pitcher_ball_rate")
    strike = rate("asof_pitcher_strike_rate")
    m1 = rate("asof_pitcher_prev1_game_middle_rate")
    m3 = rate("asof_pitcher_prev3_game_middle_rate")
    m5 = rate("asof_pitcher_prev5_game_middle_rate")
    pn = np.maximum(num("asof_pitcher_n"), 0.0)

    three_ball = (balls == 3).astype(np.float64)
    full_count = ((balls == 3) & (strikes == 2)).astype(np.float64)
    hitter_count = (balls > strikes).astype(np.float64)
    behind_count = ((balls >= 2) & (balls > strikes)).astype(np.float64)
    deep_count = ((balls + strikes) >= 4).astype(np.float64)
    two_strike = (strikes == 2).astype(np.float64)
    late_inning = (inning >= 7).astype(np.float64)
    very_late = (inning >= 9).astype(np.float64)
    close_game = (np.abs(score_diff) <= 2).astype(np.float64)
    trailing = (score_diff < 0).astype(np.float64)
    high_li = (li >= 1.5).astype(np.float64)
    extreme_li = (li >= 2.5).astype(np.float64)

    recent_success = 0.55 * p1 + 0.30 * p3 + 0.15 * p5
    recent_middle = 0.55 * m1 + 0.30 * m3 + 0.15 * m5
    success_drop = pitcher - recent_success
    command_bad = ball + middle + reverse - strike
    recent_command_bad = recent_middle - middle + ball - strike
    danger = middle + reverse
    fatigue_proxy = np.log1p(pn) * (1.0 + 0.12 * np.maximum(inning - 5.0, 0.0)) * (1.0 + 0.08 * np.maximum(month - 6.0, 0.0))
    pressure = np.log1p(li) * (1.0 + risp + 0.7 * loaded + 0.35 * runners)
    meltdown_context = pressure * (1.0 + three_ball + full_count + late_inning + close_game)

    out = pd.DataFrame(index=df.index)
    out["down_three_ball"] = three_ball
    out["down_full_count"] = full_count
    out["down_hitter_count"] = hitter_count
    out["down_behind_count"] = behind_count
    out["down_deep_count"] = deep_count
    out["down_two_strike_relief"] = two_strike
    out["down_count_risk_score"] = balls - 0.65 * strikes + 0.8 * three_ball + 1.1 * full_count
    out["down_late_inning"] = late_inning
    out["down_very_late"] = very_late
    out["down_close_late"] = late_inning * close_game
    out["down_trailing_pressure"] = trailing * pressure
    out["down_risp"] = risp
    out["down_loaded"] = loaded
    out["down_high_li"] = high_li
    out["down_extreme_li"] = extreme_li
    out["down_pressure"] = pressure
    out["down_meltdown_context"] = meltdown_context
    out["down_command_bad"] = command_bad
    out["down_danger_rate"] = danger
    out["down_recent_success_drop"] = success_drop
    out["down_recent_command_bad"] = recent_command_bad
    out["down_recent_middle_jump"] = recent_middle - middle
    out["down_fatigue_proxy"] = fatigue_proxy
    out["down_fatigue_late"] = fatigue_proxy * late_inning
    out["down_fatigue_pressure"] = fatigue_proxy * pressure
    out["down_fatigue_three_ball"] = fatigue_proxy * three_ball
    out["down_command_three_ball"] = command_bad * three_ball
    out["down_command_full_count"] = command_bad * full_count
    out["down_command_pressure"] = command_bad * pressure
    out["down_danger_pressure"] = danger * pressure
    out["down_drop_pressure"] = success_drop * pressure
    out["down_drop_late"] = success_drop * late_inning
    out["down_drop_three_ball"] = success_drop * three_ball
    out["down_pitcher_low_success_risk"] = (global_mean - pitcher)
    out["down_low_success_pressure"] = (global_mean - pitcher) * pressure
    out["down_unstable_pitcher"] = 1.0 / np.sqrt(pn + 1.0)
    out["down_unstable_pressure"] = out["down_unstable_pitcher"] * pressure
    out["down_month_fatigue"] = np.maximum(month - 5.0, 0.0) * np.log1p(pn)
    out["down_late_month_three_ball"] = np.maximum(month - 7.0, 0.0) * three_ball
    out["down_baseout_pressure"] = (runners + 1.4 * risp + 0.8 * loaded + 0.2 * outs) * np.log1p(li)
    out["down_count_pressure"] = out["down_count_risk_score"] * pressure
    out["down_count_late"] = out["down_count_risk_score"] * late_inning
    out["down_full_count_pressure"] = full_count * pressure
    out["down_three_ball_pressure"] = three_ball * pressure
    out["down_close_full_count"] = close_game * full_count
    out["down_close_three_ball"] = close_game * three_ball
    out["down_close_loaded"] = close_game * loaded
    out["down_late_loaded"] = late_inning * loaded
    out["down_late_risp"] = late_inning * risp
    out["down_late_high_li"] = late_inning * high_li
    out["down_adverse_stack"] = (
        three_ball + full_count + hitter_count + late_inning + high_li + risp + loaded + close_game
    )
    out["down_adverse_stack_sq"] = out["down_adverse_stack"] ** 2
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
