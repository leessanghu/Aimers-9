from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PersonaSpec:
    name: str
    group_cols: tuple[str, ...]
    parent_cols: tuple[str, ...]
    prior: float
    anchor: str


ALL_PERSONA_SPECS = (
    PersonaSpec("batter_vs_hand", ("batter_id", "pitcher_hand"), ("batter_id",), 45.0, "batter"),
    PersonaSpec("batter_count", ("batter_id", "persona_count"), ("batter_id",), 70.0, "batter"),
    PersonaSpec("pitcher_count", ("pitcher_id", "persona_count"), ("pitcher_id",), 55.0, "pitcher"),
    PersonaSpec("batter_count_side", ("batter_id", "persona_count_class", "pitcher_hand"), ("batter_id",), 95.0, "batter"),
    PersonaSpec("pitcher_count_side", ("pitcher_id", "persona_count_class", "batter_hand"), ("pitcher_id",), 85.0, "pitcher"),
    PersonaSpec("batter_advantage", ("batter_id", "persona_count_advantage"), ("batter_id",), 80.0, "batter"),
    PersonaSpec("pitcher_advantage", ("pitcher_id", "persona_count_advantage"), ("pitcher_id",), 70.0, "pitcher"),
    PersonaSpec("batter_twostrike", ("batter_id", "persona_two_strike"), ("batter_id",), 75.0, "batter"),
    PersonaSpec("pitcher_twostrike", ("pitcher_id", "persona_two_strike"), ("pitcher_id",), 60.0, "pitcher"),
    PersonaSpec("batter_threeball", ("batter_id", "persona_three_ball"), ("batter_id",), 90.0, "batter"),
    PersonaSpec("pitcher_threeball", ("pitcher_id", "persona_three_ball"), ("pitcher_id",), 75.0, "pitcher"),
    PersonaSpec("pitcher_pressure", ("pitcher_id", "persona_pressure"), ("pitcher_id",), 70.0, "pitcher"),
    PersonaSpec("pitcher_inning", ("pitcher_id", "persona_inning"), ("pitcher_id",), 60.0, "pitcher"),
    PersonaSpec("batter_inning", ("batter_id", "persona_inning"), ("batter_id",), 75.0, "batter"),
    PersonaSpec("pitcher_late_close", ("pitcher_id", "persona_late_close"), ("pitcher_id",), 85.0, "pitcher"),
    PersonaSpec("batter_late_close", ("batter_id", "persona_late_close"), ("batter_id",), 95.0, "batter"),
    PersonaSpec("pitcher_opponent", ("pitcher_id", "batter_team_id"), ("pitcher_id",), 65.0, "pitcher"),
    PersonaSpec("pitcher_baseout", ("pitcher_id", "persona_baseout"), ("pitcher_id",), 90.0, "pitcher"),
    PersonaSpec("batter_baseout", ("batter_id", "persona_baseout"), ("batter_id",), 105.0, "batter"),
    PersonaSpec("pitcher_runner_threat", ("pitcher_id", "persona_runner_threat"), ("pitcher_id",), 75.0, "pitcher"),
    PersonaSpec("batter_runner_threat", ("batter_id", "persona_runner_threat"), ("batter_id",), 90.0, "batter"),
    PersonaSpec("pitcher_li_bucket", ("pitcher_id", "persona_li_bucket"), ("pitcher_id",), 80.0, "pitcher"),
    PersonaSpec("batter_li_bucket", ("batter_id", "persona_li_bucket"), ("batter_id",), 95.0, "batter"),
    PersonaSpec("pitcher_score_bucket", ("pitcher_id", "persona_score_bucket"), ("pitcher_id",), 85.0, "pitcher"),
    PersonaSpec("batter_score_bucket", ("batter_id", "persona_score_bucket"), ("batter_id",), 95.0, "batter"),
    PersonaSpec("pitcher_month", ("pitcher_id", "game_month"), ("pitcher_id",), 65.0, "pitcher"),
    PersonaSpec("batter_month", ("batter_id", "game_month"), ("batter_id",), 85.0, "batter"),
    PersonaSpec("pitcher_daytype", ("pitcher_id", "persona_daytype"), ("pitcher_id",), 85.0, "pitcher"),
    PersonaSpec("batter_daytype", ("batter_id", "persona_daytype"), ("batter_id",), 95.0, "batter"),
    PersonaSpec("direct_matchup", ("pitcher_id", "batter_id"), ("pitcher_id",), 120.0, "pitcher"),
    PersonaSpec("direct_matchup_countclass", ("pitcher_id", "batter_id", "persona_count_class"), ("pitcher_id",), 180.0, "pitcher"),
    PersonaSpec("direct_matchup_pressure", ("pitcher_id", "batter_id", "persona_runner_threat"), ("pitcher_id",), 200.0, "pitcher"),
    PersonaSpec(
        "team_count_hand",
        ("pitcher_team_id", "persona_count", "batter_hand"),
        ("pitcher_team_id",),
        120.0,
        "pitcher",
    ),
    PersonaSpec(
        "team_advantage_hand",
        ("pitcher_team_id", "persona_count_advantage", "batter_hand"),
        ("pitcher_team_id",),
        130.0,
        "pitcher",
    ),
    PersonaSpec(
        "offense_advantage_side",
        ("batter_team_id", "persona_count_advantage", "pitcher_hand"),
        ("batter_team_id",),
        145.0,
        "batter",
    ),
    PersonaSpec("batter_pressure", ("batter_id", "persona_pressure"), ("batter_id",), 85.0, "batter"),
    PersonaSpec("batter_opponent", ("batter_id", "pitcher_team_id"), ("batter_id",), 90.0, "batter"),
    PersonaSpec("platoon_count", ("pitcher_hand", "batter_hand", "persona_count_class"), ("pitcher_hand", "batter_hand"), 220.0, "pitcher"),
    PersonaSpec("baseout_count", ("persona_baseout", "persona_count_class"), ("persona_baseout",), 260.0, "pitcher"),
)

# Keep the active set focused enough to validate quickly. The excluded specs remain
# available in ALL_PERSONA_SPECS for slower follow-up sweeps.
PERSONA_SPECS = tuple(
    spec
    for spec in ALL_PERSONA_SPECS
    if spec.name
    in {
        "batter_vs_hand",
        "batter_count",
        "pitcher_count",
        "pitcher_pressure",
        "pitcher_inning",
        "pitcher_opponent",
        "pitcher_baseout",
        "pitcher_month",
        "direct_matchup",
        "team_count_hand",
        "batter_pressure",
    }
)


def add_persona_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["persona_count"] = out["balls_before"].astype("int16") * 3 + out["strikes_before"].astype("int16")
    balls = out["balls_before"].astype("int16")
    strikes = out["strikes_before"].astype("int16")
    out["persona_count_advantage"] = np.select(
        [strikes > balls, balls > strikes],
        [0, 2],
        default=1,
    ).astype("int8")
    out["persona_count_class"] = np.select(
        [
            (balls == 0) & (strikes == 0),
            strikes >= 2,
            balls >= 3,
            balls > strikes,
            strikes > balls,
        ],
        [0, 1, 2, 3, 4],
        default=5,
    ).astype("int8")
    out["persona_two_strike"] = (strikes >= 2).astype("int8")
    out["persona_three_ball"] = (balls >= 3).astype("int8")
    risp = ((out["runner_on_2b"] == 1) | (out["runner_on_3b"] == 1)).astype("int8")
    high_li = (out["li"] >= 1.5).astype("int8")
    late = (out["inning"] >= 7).astype("int8")
    out["persona_pressure"] = high_li + 2 * risp + 4 * late
    out["persona_li_bucket"] = np.select(
        [out["li"] < 0.7, out["li"] < 1.2, out["li"] < 2.0],
        [0, 1, 2],
        default=3,
    ).astype("int8")
    out["persona_inning"] = np.select(
        [out["inning"] <= 3, out["inning"] <= 6, out["inning"] <= 9],
        [0, 1, 2],
        default=3,
    ).astype("int8")
    out["persona_late_close"] = ((out["inning"] >= 7) & (out["score_diff_pitcher_team"].abs() <= 2)).astype("int8")
    out["persona_score_bucket"] = np.select(
        [
            out["score_diff_pitcher_team"] <= -3,
            out["score_diff_pitcher_team"] < 0,
            out["score_diff_pitcher_team"] == 0,
            out["score_diff_pitcher_team"] <= 2,
        ],
        [0, 1, 2, 3],
        default=4,
    ).astype("int8")
    out["persona_runner_threat"] = np.select(
        [
            out["num_runners_on"] == 0,
            (out["runner_on_1b"] == 1) & (out["num_runners_on"] == 1),
            risp == 1,
        ],
        [0, 1, 2],
        default=3,
    ).astype("int8")
    out["persona_daytype"] = (out["game_dayofweek"] >= 5).astype("int8")
    out["persona_baseout"] = (
        out["runner_on_1b"].astype("int8")
        + 2 * out["runner_on_2b"].astype("int8")
        + 4 * out["runner_on_3b"].astype("int8")
        + 8 * out["outs_before"].astype("int8")
    )
    return out


def _aggregate_history(
    keyed: pd.DataFrame,
    target: np.ndarray,
    cols: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = keyed[list(cols) + ["season"]].copy()
    work["_y"] = np.asarray(target, dtype=np.float32)
    yearly = work.groupby([*cols, "season"], observed=True, sort=False)["_y"].agg(["sum", "count"]).reset_index()
    yearly = yearly.sort_values([*cols, "season"])
    grouped = yearly.groupby(list(cols), observed=True, sort=False)
    yearly["hist_sum"] = grouped["sum"].cumsum() - yearly["sum"]
    yearly["hist_n"] = grouped["count"].cumsum() - yearly["count"]
    final = yearly.groupby(list(cols), observed=True, sort=False)[["sum", "count"]].sum().reset_index()
    final = final.rename(columns={"sum": "hist_sum", "count": "hist_n"})
    return yearly[[*cols, "season", "hist_sum", "hist_n"]], final


def _feature_frame(
    base: pd.DataFrame,
    group_sum: np.ndarray,
    group_n: np.ndarray,
    parent_sum: np.ndarray,
    parent_n: np.ndarray,
    global_mean: float,
    spec: PersonaSpec,
) -> pd.DataFrame:
    parent_rate = (parent_sum + 100.0 * global_mean) / (parent_n + 100.0)
    rate = (group_sum + spec.prior * parent_rate) / (group_n + spec.prior)
    reliability = group_n / (group_n + spec.prior)
    uncertainty = np.sqrt(np.clip(rate * (1.0 - rate) / (group_n + spec.prior + 1.0), 0.0, None))
    if spec.anchor == "pitcher":
        anchor = pd.to_numeric(base["asof_pitcher_success_rate"], errors="coerce").fillna(global_mean).to_numpy()
    else:
        anchor = pd.to_numeric(base["asof_batter_success_rate"], errors="coerce").fillna(global_mean).to_numpy()
    prefix = f"persona_{spec.name}"
    return pd.DataFrame(
        {
            f"{prefix}_rate": rate.astype("float32"),
            f"{prefix}_delta": (rate - parent_rate).astype("float32"),
            f"{prefix}_log_n": np.log1p(group_n).astype("float32"),
            f"{prefix}_reliability": reliability.astype("float32"),
            f"{prefix}_uncertainty": uncertainty.astype("float32"),
            f"{prefix}_vs_asof": (rate - anchor).astype("float32"),
        },
        index=base.index,
    )


def fit_transform_personas(
    train_df: pd.DataFrame,
    target: np.ndarray,
) -> tuple[pd.DataFrame, dict]:
    keyed = add_persona_keys(train_df)
    global_mean = float(np.mean(target))
    features = []
    states: dict[str, dict] = {}

    cache: dict[tuple[str, ...], tuple[pd.DataFrame, pd.DataFrame]] = {}
    needed = {spec.group_cols for spec in PERSONA_SPECS} | {spec.parent_cols for spec in PERSONA_SPECS}
    for cols in needed:
        cache[cols] = _aggregate_history(keyed, target, cols)

    for spec in PERSONA_SPECS:
        group_yearly, group_final = cache[spec.group_cols]
        parent_yearly, parent_final = cache[spec.parent_cols]
        group_cols = list(spec.group_cols)
        parent_cols = list(spec.parent_cols)
        row_cols = list(dict.fromkeys(group_cols + parent_cols + ["season"]))
        row = keyed[row_cols].copy()
        row["_row_order"] = np.arange(len(row))
        row = row.merge(group_yearly, on=group_cols + ["season"], how="left", sort=False)
        row = row.rename(columns={"hist_sum": "group_sum", "hist_n": "group_n"})
        row = row.merge(parent_yearly, on=parent_cols + ["season"], how="left", sort=False)
        row = row.rename(columns={"hist_sum": "parent_sum", "hist_n": "parent_n"}).sort_values("_row_order")
        for col in ["group_sum", "group_n", "parent_sum", "parent_n"]:
            row[col] = row[col].fillna(0.0)
        features.append(
            _feature_frame(
                train_df,
                row["group_sum"].to_numpy(),
                row["group_n"].to_numpy(),
                row["parent_sum"].to_numpy(),
                row["parent_n"].to_numpy(),
                global_mean,
                spec,
            )
        )
        states[spec.name] = {"group": group_final, "parent": parent_final}

    state = {"global_mean": global_mean, "specs": PERSONA_SPECS, "tables": states}
    return pd.concat(features, axis=1), state


def transform_personas(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    keyed = add_persona_keys(df)
    global_mean = float(state["global_mean"])
    features = []
    for spec in state["specs"]:
        tables = state["tables"][spec.name]
        group_cols = list(spec.group_cols)
        parent_cols = list(spec.parent_cols)
        row_cols = list(dict.fromkeys(group_cols + parent_cols))
        row = keyed[row_cols].copy()
        row["_row_order"] = np.arange(len(row))
        row = row.merge(tables["group"], on=group_cols, how="left", sort=False)
        row = row.rename(columns={"hist_sum": "group_sum", "hist_n": "group_n"})
        row = row.merge(tables["parent"], on=parent_cols, how="left", sort=False)
        row = row.rename(columns={"hist_sum": "parent_sum", "hist_n": "parent_n"}).sort_values("_row_order")
        for col in ["group_sum", "group_n", "parent_sum", "parent_n"]:
            row[col] = row[col].fillna(0.0)
        features.append(
            _feature_frame(
                df,
                row["group_sum"].to_numpy(),
                row["group_n"].to_numpy(),
                row["parent_sum"].to_numpy(),
                row["parent_n"].to_numpy(),
                global_mean,
                spec,
            )
        )
    return pd.concat(features, axis=1)
