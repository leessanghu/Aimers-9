from __future__ import annotations

import numpy as np
import pandas as pd


def add_orthogonal_features(df: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    eps = 1e-6

    def num(name: str, fill: float = 0.0) -> np.ndarray:
        return pd.to_numeric(df[name], errors="coerce").fillna(fill).to_numpy(np.float64)

    def rate(name: str, fill: float | None = None) -> np.ndarray:
        if fill is None:
            fill = global_mean
        return pd.to_numeric(df[name], errors="coerce").fillna(fill).to_numpy(np.float64)

    pitcher = np.clip(rate("asof_pitcher_success_rate"), eps, 1.0 - eps)
    batter = np.clip(rate("asof_batter_success_rate"), eps, 1.0 - eps)
    middle = np.clip(rate("asof_pitcher_middle_rate"), eps, 1.0 - eps)
    reverse = np.clip(rate("asof_pitcher_reverse_rate"), eps, 1.0 - eps)
    ball = np.clip(rate("asof_pitcher_ball_rate"), eps, 1.0 - eps)
    strike = np.clip(rate("asof_pitcher_strike_rate"), eps, 1.0 - eps)
    batter_middle = np.clip(rate("asof_batter_middle_rate"), eps, 1.0 - eps)
    fast = np.clip(rate("asof_pitcher_fastball_rate", 1.0 / 3.0), eps, 1.0)
    breaking = np.clip(rate("asof_pitcher_breaking_rate", 1.0 / 3.0), eps, 1.0)
    offspeed = np.clip(rate("asof_pitcher_offspeed_rate", 1.0 / 3.0), eps, 1.0)
    mix_sum = np.maximum(fast + breaking + offspeed, eps)
    fast = fast / mix_sum
    breaking = breaking / mix_sum
    offspeed = offspeed / mix_sum

    pn = np.maximum(num("asof_pitcher_n"), 0.0)
    bn = np.maximum(num("asof_batter_n"), 0.0)
    mix_n = np.maximum(num("asof_pitcher_pitchmix_n"), 0.0)
    balls = num("balls_before")
    strikes = num("strikes_before")
    outs = num("outs_before")
    inning = num("inning", 5.0)
    month = num("game_month", 6.0)
    li = np.maximum(num("li", 1.0), 0.0)
    hwe = np.clip(num("home_win_expectancy", 0.5), eps, 1.0 - eps)
    awe = np.clip(num("away_win_expectancy", 0.5), eps, 1.0 - eps)
    score_diff = num("score_diff_pitcher_team")
    runners = num("num_runners_on")
    r2 = num("runner_on_2b")
    r3 = num("runner_on_3b")
    risp = ((r2 > 0) | (r3 > 0)).astype(np.float64)

    p1 = rate("asof_pitcher_prev1_game_success_rate")
    p3 = rate("asof_pitcher_prev3_game_success_rate")
    p5 = rate("asof_pitcher_prev5_game_success_rate")
    m1 = rate("asof_pitcher_prev1_game_middle_rate")
    m3 = rate("asof_pitcher_prev3_game_middle_rate")
    m5 = rate("asof_pitcher_prev5_game_middle_rate")

    mix_entropy = -(fast * np.log(fast) + breaking * np.log(breaking) + offspeed * np.log(offspeed))
    mix_entropy = mix_entropy / np.log(3.0)
    mix_concentration = fast**2 + breaking**2 + offspeed**2
    mix_breaking_offspeed = breaking + offspeed
    mix_fast_minus_soft = fast - mix_breaking_offspeed
    command_logit = np.log(strike / (1.0 - strike)) - np.log(ball / (1.0 - ball))
    danger_logit = np.log((middle + reverse + eps) / np.maximum(1.0 - middle - reverse, eps))
    pitcher_uncert = np.sqrt(pitcher * (1.0 - pitcher) / (pn + 30.0))
    batter_uncert = np.sqrt(batter * (1.0 - batter) / (bn + 30.0))
    mix_uncert = 1.0 / np.sqrt(mix_n + 10.0)
    recent_success_curve = (p1 - p3) + 0.5 * (p3 - p5)
    recent_middle_curve = (m1 - m3) + 0.5 * (m3 - m5)
    count_severity = balls - strikes + 0.75 * ((balls == 3) & (strikes == 2)) + 0.4 * (balls == 3)
    leverage_asym = np.log1p(li) * np.sign(score_diff) * np.sqrt(np.abs(score_diff) + 1.0)
    winexp_spread = hwe - awe
    winexp_entropy = -(hwe * np.log(hwe) + awe * np.log(awe))

    out = pd.DataFrame(index=df.index)
    out["ort_mix_entropy"] = mix_entropy
    out["ort_mix_concentration"] = mix_concentration
    out["ort_mix_fast_minus_soft"] = mix_fast_minus_soft
    out["ort_mix_breaking_offspeed"] = mix_breaking_offspeed
    out["ort_mix_uncertainty"] = mix_uncert
    out["ort_mix_entropy_uncert"] = mix_entropy * mix_uncert
    out["ort_command_logit"] = command_logit
    out["ort_danger_logit"] = danger_logit
    out["ort_command_minus_danger"] = command_logit - danger_logit
    out["ort_pitcher_uncertainty"] = pitcher_uncert
    out["ort_batter_uncertainty"] = batter_uncert
    out["ort_uncertainty_gap"] = pitcher_uncert - batter_uncert
    out["ort_low_sample_matchup"] = 1.0 / np.sqrt(np.minimum(pn, bn) + 1.0)
    out["ort_recent_success_curve"] = recent_success_curve
    out["ort_recent_middle_curve"] = recent_middle_curve
    out["ort_recent_curve_conflict"] = recent_success_curve - recent_middle_curve
    out["ort_count_severity"] = count_severity
    out["ort_count_command"] = count_severity * command_logit
    out["ort_count_danger"] = count_severity * danger_logit
    out["ort_count_mix_soft"] = count_severity * mix_breaking_offspeed
    out["ort_late_mix_entropy"] = (inning >= 7).astype(np.float64) * mix_entropy
    out["ort_late_command"] = (inning >= 7).astype(np.float64) * command_logit
    out["ort_month_mix_uncert"] = np.maximum(month - 5.0, 0.0) * mix_uncert
    out["ort_month_command_decay"] = np.maximum(month - 5.0, 0.0) * recent_success_curve
    out["ort_leverage_asym"] = leverage_asym
    out["ort_winexp_spread"] = winexp_spread
    out["ort_winexp_entropy"] = winexp_entropy
    out["ort_li_winexp_entropy"] = np.log1p(li) * winexp_entropy
    out["ort_risp_command"] = risp * command_logit
    out["ort_risp_danger"] = risp * danger_logit
    out["ort_runners_mix_soft"] = runners * mix_breaking_offspeed
    out["ort_outs_count_severity"] = outs * count_severity
    out["ort_batter_middle_vs_pitcher_danger"] = batter_middle - (middle + reverse)
    out["ort_batter_middle_count"] = batter_middle * count_severity
    out["ort_batter_middle_pressure"] = batter_middle * np.log1p(li) * (1.0 + risp)
    out["ort_pitcher_batter_uncert_product"] = pitcher_uncert * batter_uncert
    out["ort_style_risk"] = mix_breaking_offspeed * batter_middle + fast * ball
    out["ort_style_command_gap"] = mix_fast_minus_soft * command_logit
    out["ort_high_entropy_pressure"] = mix_entropy * np.log1p(li) * (1.0 + risp)
    out["ort_low_entropy_count"] = (1.0 - mix_entropy) * count_severity
    out["ort_command_reliability"] = command_logit * np.sqrt(pn / (pn + 200.0))
    out["ort_danger_reliability"] = danger_logit * np.sqrt(pn / (pn + 200.0))
    out["ort_style_reliability"] = (mix_concentration - 1.0 / 3.0) * np.sqrt(mix_n / (mix_n + 100.0))
    out["ort_leverage_style"] = np.log1p(li) * (mix_entropy - mix_concentration)
    return out.replace([np.inf, -np.inf], 0.0).fillna(0.0).astype(np.float32)
