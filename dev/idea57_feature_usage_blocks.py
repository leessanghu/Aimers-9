"""Aggregate existing v35 SHAP/split usage into semantic feature blocks.

Individual importance is unstable under correlated proxies.  Block sums are the
right diagnostic for the question 'what information family consumes capacity?'.
"""
from __future__ import annotations

import pandas as pd


def block(name: str) -> str:
    # Order matters: reliability and batter fields must be caught before broad
    # pitcher/ability patterns.
    reliability_exact = {
        "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n", "inseason_n",
        "platoon_n", "inning_n", "pt_n", "count_n", "ly_n", "bat_inseason_n",
        "bat_ly_n", "bplatoon_n", "role_n_app", "vol_n_seasons",
        "pitcher_id_count", "batter_id_count", "pitcher_team_id_count", "batter_team_id_count",
    }
    if name in reliability_exact or name.startswith("flag_") or name.endswith("_missing"):
        return "sample_reliability"
    if name.startswith("tm_"):
        return "trackman"
    if (name.startswith("bat_") or name.startswith("batter_") or name.startswith("asof_batter")
            or name.startswith("bplatoon") or name in {"x_p_over_b", "x_platoon_x_samehand"}):
        return "batter_matchup"
    context_exact = {
        "cat_top_bottom", "cat_base_state", "inning", "balls_before", "strikes_before",
        "outs_before", "run_top_before", "run_bot_before", "run_total_before",
        "score_diff_home", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b",
        "runner_on_3b", "num_runners_on", "home_win_expectancy", "away_win_expectancy",
        "li", "pitcher_hand", "batter_hand", "same_hand", "count_state", "hand_matchup",
        "count_diff", "x_count_pressure", "role_x_inning",
    }
    if name in context_exact or name.startswith("x_ability_x_count") or name.startswith("x_ability_x_pressure"):
        return "game_context"
    if (name in {"season", "game_month", "game_dayofweek", "cat_game_type"}
            or "team_id" in name):
        return "environment_team"
    ability_prefixes = (
        "asof_pitcher_", "inseason_", "x_ability", "ability_", "platoon_", "inning_",
        "pt_", "ly_", "vol_", "role_", "form", "diff_", "x_exp", "x_rev", "x_mid",
        "x_kal", "x_prev", "x_ball_over_strike",
    )
    if name.startswith(ability_prefixes):
        return "pitcher_ability"
    return "other"


shap = pd.read_csv("phase94_shap_magnitude.csv")
shap.columns = ["feature", "shap"]
splits = pd.read_csv("phase94_split_counts.csv")
splits.columns = ["feature", "splits"]
d = shap.merge(splits, on="feature", how="left").fillna({"splits": 0})
d["block"] = d["feature"].map(block)
d["shap_share"] = d["shap"] / d["shap"].sum()
d["split_share"] = d["splits"] / d["splits"].sum()

summary = d.groupby("block").agg(
    n_features=("feature", "size"), shap=("shap", "sum"), shap_share=("shap_share", "sum"),
    splits=("splits", "sum"), split_share=("split_share", "sum"),
).sort_values("shap_share", ascending=False)
summary.to_csv("idea57_feature_block_usage.csv")
d.sort_values(["block", "shap"], ascending=[True, False]).to_csv("idea57_feature_usage_detail.csv", index=False)

print("v35 CatBoost feature-block usage")
print(summary.to_string(formatters={"shap_share": "{:.2%}".format, "split_share": "{:.2%}".format}))
print("\ntop features per block")
for b, g in d.groupby("block"):
    top = g.nlargest(5, "shap")[["feature", "shap_share", "splits"]]
    print(f"\n[{b}]")
    print(top.to_string(index=False, formatters={"shap_share": "{:.2%}".format}))
