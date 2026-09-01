"""커리어 시즌별 변동성 — AMEX/Home Credit류 "entity 여러 과거 관측치를 std/min/max로 압축"
패턴을 그대로 적용. 지금까지는 '평균 제구력'만 썼지, 이 투수가 시즌마다 얼마나 들쭉날쭉한지는
피처로 만든 적이 없다.

만드는 값: 그 행의 '직전 시즌까지' 있는 모든 과거 시즌들의 시즌별(고립된) 성공률에 대해
  vol_std : 시즌간 성공률 표준편차 (일관성)
  vol_min : 최저 시즌 성공률 (바닥)
  vol_max : 최고 시즌 성공률 (천장)
  vol_range: max-min
  vol_n_seasons: 관측된 과거 시즌 수

leakage 안전성: 시즌별 고립 성공률은 season_end_table(각 시즌 종료 시점 누적)의 연속 시즌
차분으로만 계산하고(다른 투수/다른 행 참조 없음), 각 행은 자기 투수의 '직전 시즌까지' 시즌들만
집계에 포함한다 (expanding 후 1칸 shift로 자기 시즌 자체는 제외).
"""

import numpy as np
import pandas as pd

VOL_COLS = ["vol_std", "vol_min", "vol_max", "vol_range", "vol_n_seasons"]
K_VOL = 3.0  # vol_std를 전역 평균 변동성 쪽으로 축소 (관측 시즌 수 적을 때)


def build_season_isolated_rates(season_end_table):
    """season_end_table((pitcher,season)->누적 N_end/S_end)에서 연속 시즌 차분으로
    '그 시즌만의' 고립 성공률을 만든다."""
    t = season_end_table.sort_values(["pitcher_id", "season"]).copy()
    g = t.groupby("pitcher_id")
    t["N_prev"] = g["N_end"].shift(1).fillna(0.0)
    t["S_prev"] = g["S_end"].shift(1).fillna(0.0)
    t["n_season"] = (t["N_end"] - t["N_prev"]).clip(lower=0)
    t["s_season"] = (t["S_end"] - t["S_prev"]).clip(lower=0)
    t["rate_season"] = np.where(t["n_season"] > 0, t["s_season"] / t["n_season"], np.nan)
    return t[["pitcher_id", "season", "rate_season", "n_season"]]


def build_volatility_table(season_end_table, min_n_season=10.0):
    """유효한(그 시즌 표본이 충분한) 시즌 rate만 가지고, 각 (pitcher,season)에 대해
    '이 시즌까지 포함한' expanding std/min/max를 만든다. transform에서 1칸 밀어 조회한다."""
    iso = build_season_isolated_rates(season_end_table)
    iso.loc[iso["n_season"] < min_n_season, "rate_season"] = np.nan  # 표본 너무 적은 시즌은 변동성 계산에서 제외

    iso = iso.sort_values(["pitcher_id", "season"])
    grp = iso.groupby("pitcher_id")["rate_season"]
    iso["exp_std"] = grp.expanding().std().reset_index(level=0, drop=True)
    iso["exp_min"] = grp.expanding().min().reset_index(level=0, drop=True)
    iso["exp_max"] = grp.expanding().max().reset_index(level=0, drop=True)
    iso["exp_count"] = grp.apply(lambda s: s.notna().expanding().sum()).reset_index(level=0, drop=True)
    return iso[["pitcher_id", "season", "exp_std", "exp_min", "exp_max", "exp_count"]]


def _fixed_global_std(vol_table):
    """vol_table(=fit/train에서 만든 표) 자체에서 한 번만 계산하는 고정 상수.
    transform 호출 시점의 배치(subset이든 full이든)에 따라 달라지면 행 독립성 규칙 위반이라
    반드시 fit 시점 값을 export_stats로 고정해 재사용해야 한다."""
    std = vol_table["exp_std"].to_numpy(np.float64)
    count = vol_table["exp_count"].to_numpy(np.float64)
    valid = (count >= 2) & ~np.isnan(std)
    return float(np.mean(std[valid])) if valid.any() else 0.05


def transform_volatility(df, vol_table, seasons_range, k=K_VOL, global_std=None):
    if global_std is None:
        global_std = _fixed_global_std(vol_table)

    cols = ["exp_std", "exp_min", "exp_max", "exp_count"]
    piv = {c: vol_table.pivot(index="pitcher_id", columns="season", values=c)
                       .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
           for c in cols}
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    vals = {c: piv[c].reindex(idx).to_numpy().astype(np.float64) for c in cols}

    exp_count = np.nan_to_num(vals["exp_count"], nan=0.0)
    exp_std = vals["exp_std"]
    exp_min = vals["exp_min"]
    exp_max = vals["exp_max"]

    valid = exp_count >= 2
    shrunk_std = np.where(valid, (exp_count * np.nan_to_num(exp_std, nan=global_std) + k * global_std) / (exp_count + k), global_std)

    out = pd.DataFrame(index=df.index)
    out["vol_std"] = shrunk_std
    out["vol_min"] = np.where(exp_count >= 1, np.nan_to_num(exp_min, nan=0.5), 0.5)
    out["vol_max"] = np.where(exp_count >= 1, np.nan_to_num(exp_max, nan=0.5), 0.5)
    out["vol_range"] = out["vol_max"] - out["vol_min"]
    out["vol_n_seasons"] = np.log1p(exp_count)
    return out


def export_stats(vol_table, seasons_range, k=K_VOL):
    return {"vol_table": vol_table, "seasons_range": list(seasons_range), "k": float(k),
            "global_std": _fixed_global_std(vol_table)}
