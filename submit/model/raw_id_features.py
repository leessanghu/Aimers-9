from __future__ import annotations

import numpy as np
import pandas as pd

from advanced_features import add_advanced_features


BASE_NUMERIC = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

CAT_COLS = [
    "cat_pitcher_id",
    "cat_batter_id",
    "cat_pitcher_team_id",
    "cat_batter_team_id",
    "cat_pitcher_hand",
    "cat_batter_hand",
    "cat_count",
    "cat_count_class",
    "cat_baseout",
    "cat_month",
    "cat_inning_bucket",
    "cat_phase",
    "cat_pitcher_batter",
    "cat_pitcher_count",
    "cat_batter_count",
    "cat_pitcher_month",
    "cat_batter_month",
    "cat_pitcher_baseout",
    "cat_batter_baseout",
    "cat_pitcher_batter_count",
    "cat_pitcher_batter_hand",
    "cat_team_count_hand",
]


def id_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    balls = df["balls_before"].astype("int16")
    strikes = df["strikes_before"].astype("int16")
    count = balls.astype(str) + "-" + strikes.astype(str)
    count_class = np.select(
        [
            (balls == 0) & (strikes == 0),
            strikes >= 2,
            balls >= 3,
            balls > strikes,
            strikes > balls,
        ],
        ["first", "two_strike", "three_ball", "hitter", "pitcher"],
        default="even",
    )
    baseout = (
        df["runner_on_1b"].astype("int8").astype(str)
        + df["runner_on_2b"].astype("int8").astype(str)
        + df["runner_on_3b"].astype("int8").astype(str)
        + "_o"
        + df["outs_before"].astype("int8").astype(str)
    )
    inning_bucket = pd.cut(
        df["inning"].clip(1, 12),
        bins=[0, 3, 6, 9, 99],
        labels=["early", "mid", "late", "extras"],
        include_lowest=True,
    ).astype(str)
    phase_name = pd.Series(
        np.select([df["game_month"] <= 4, df["game_month"] <= 7], ["early", "middle"], "late"),
        index=df.index,
    )

    pid = df["pitcher_id"].astype(str)
    bid = df["batter_id"].astype(str)
    ph = df["pitcher_hand"].astype(str)
    bh = df["batter_hand"].astype(str)
    pt = df["pitcher_team_id"].astype(str)
    bt = df["batter_team_id"].astype(str)
    month = df["game_month"].astype("int16").astype(str)

    out["cat_pitcher_id"] = pid
    out["cat_batter_id"] = bid
    out["cat_pitcher_team_id"] = pt
    out["cat_batter_team_id"] = bt
    out["cat_pitcher_hand"] = ph
    out["cat_batter_hand"] = bh
    out["cat_count"] = count
    out["cat_count_class"] = count_class
    out["cat_baseout"] = baseout
    out["cat_month"] = month
    out["cat_inning_bucket"] = inning_bucket
    out["cat_phase"] = phase_name
    out["cat_pitcher_batter"] = pid + "_" + bid
    out["cat_pitcher_count"] = pid + "_" + count
    out["cat_batter_count"] = bid + "_" + count
    out["cat_pitcher_month"] = pid + "_m" + month
    out["cat_batter_month"] = bid + "_m" + month
    out["cat_pitcher_baseout"] = pid + "_" + baseout
    out["cat_batter_baseout"] = bid + "_" + baseout
    out["cat_pitcher_batter_count"] = pid + "_" + bid + "_" + count
    out["cat_pitcher_batter_hand"] = pid + "_" + bid + "_" + ph + bh
    out["cat_team_count_hand"] = pt + "_" + bt + "_" + count + "_" + ph + bh
    return out.astype(str)


def build_raw_id_matrix(df: pd.DataFrame, global_mean: float) -> pd.DataFrame:
    numeric = pd.concat(
        [
            df[BASE_NUMERIC].reset_index(drop=True),
            add_advanced_features(df, global_mean).reset_index(drop=True),
        ],
        axis=1,
    ).apply(pd.to_numeric, errors="coerce").astype(np.float32)
    cats = id_categorical_features(df).reset_index(drop=True)
    return pd.concat([numeric, cats], axis=1)


def raw_cat_indices(matrix: pd.DataFrame) -> list[int]:
    return [matrix.columns.get_loc(col) for col in CAT_COLS]
