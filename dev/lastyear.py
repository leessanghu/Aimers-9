"""'작년 한 시즌만' 피처 + 5개 rate 합성 실력 추정치.

진단(phase26): 다음 시즌 예측 R^2
  커리어 누적만 0.2821 / 작년 한 시즌만 0.2841 / 둘 다 0.3051 / +4개rate 0.3156
  -> 커리어와 작년은 보완재인데 우리 모델엔 '작년 한 시즌만'이 아예 없다.
  -> reverse_rate 상관 -0.4939 (success +0.5311에 맞먹음)인데 실력추정에 안 쓰고 있다.

복원 방식: season_end_table의 누적값 차분.  작년치 = 누적(S) - 누적(S-1)
  (in-season이 '현재 누적 - 작년말 누적'으로 +114를 낸 것과 같은 트릭의 한 칸 앞)

leakage: 각 행은 자기 투수의 season-1, season-2 시점 누적만 조회. 행 간 참조 없음.
"""

import numpy as np
import pandas as pd

TARGET = "control_success"
K_LY = 30.0   # 작년치 스무딩 (한 시즌 표본이라 in-season K=15보다 크게)

LASTYEAR_COLS = ["ly_success", "ly_reverse", "ly_ball", "ly_middle", "ly_n",
                 "ly_minus_career", "ability_composite"]

# phase26 회귀에서 얻은 방향/강도 (커리어+작년+4rate -> 다음시즌 success)
# 부호가 baseball적으로도 타당: reverse/ball/middle 높을수록 다음시즌 제구 나쁨
COMPOSITE_W = {"career": 0.45, "ly_success": 0.35, "ly_reverse": -0.20,
               "ly_ball": -0.08, "ly_middle": -0.10}


def build_lastyear_table(df):
    """(pitcher_id, season) -> 그 시즌 '끝난 시점'의 누적 (n, 성공, 역방향, 볼, 미들).

    inseason.build_season_end_table과 같은 off-by-one 보정(마지막 투구 자신 포함)."""
    if "row_num" not in df.columns:
        df = df.assign(row_num=df["row_id"].str.replace("TRAIN_", "", regex=False)
                       .str.replace("TEST_", "", regex=False).astype(int))
    sub = df.sort_values(["pitcher_id", "row_num"])
    last = sub.groupby(["pitcher_id", "season"], as_index=False).last()
    nb = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)

    out = pd.DataFrame({"pitcher_id": last["pitcher_id"], "season": last["season"]})
    out["N_end"] = nb + 1
    out["S_end"] = (np.round(last["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * nb)
                    + last[TARGET].to_numpy(np.float64))
    for col, nm in [("asof_pitcher_reverse_rate", "R_end"), ("asof_pitcher_ball_rate", "B_end"),
                    ("asof_pitcher_middle_rate", "M_end")]:
        out[nm] = np.round(last[col].fillna(0).to_numpy(np.float64) * nb)
    return out


def transform_lastyear(df, ly_table, global_rates, seasons_range, k=K_LY):
    """작년 한 시즌만의 rate들 + 합성 실력 추정치."""
    cols = ["N_end", "S_end", "R_end", "B_end", "M_end"]
    pivots = {c: ly_table.pivot(index="pitcher_id", columns="season", values=c)
                          .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
              for c in cols}

    pid = df["pitcher_id"].to_numpy()
    season = df["season"].to_numpy()
    idx_prev = pd.MultiIndex.from_arrays([pid, season - 1])   # 작년 말 누적
    idx_prev2 = pd.MultiIndex.from_arrays([pid, season - 2])  # 재작년 말 누적

    cum1, cum2 = {}, {}
    for c in cols:
        cum1[c] = np.nan_to_num(pivots[c].reindex(idx_prev).to_numpy().astype(np.float64), nan=0.0)
        cum2[c] = np.nan_to_num(pivots[c].reindex(idx_prev2).to_numpy().astype(np.float64), nan=0.0)

    n_ly = np.clip(cum1["N_end"] - cum2["N_end"], 0, None)          # 작년 한 시즌 투구수
    career_n = cum1["N_end"]
    career_rate = np.divide(cum1["S_end"], career_n, out=np.full_like(career_n, np.nan), where=career_n > 0)
    career_rate = np.nan_to_num(career_rate, nan=global_rates["success"])

    out = pd.DataFrame(index=df.index)
    for src, nm, gkey in [("S_end", "ly_success", "success"), ("R_end", "ly_reverse", "reverse"),
                          ("B_end", "ly_ball", "ball"), ("M_end", "ly_middle", "middle")]:
        cnt = np.clip(cum1[src] - cum2[src], 0, None)
        raw = np.divide(cnt, n_ly, out=np.full_like(n_ly, np.nan), where=n_ly > 0)
        gm = global_rates[gkey]
        out[nm] = (n_ly * np.nan_to_num(raw, nan=gm) + k * gm) / (n_ly + k)

    out["ly_n"] = np.log1p(n_ly)
    out["ly_minus_career"] = out["ly_success"].to_numpy() - career_rate

    comp = (COMPOSITE_W["career"] * career_rate
            + COMPOSITE_W["ly_success"] * out["ly_success"].to_numpy()
            + COMPOSITE_W["ly_reverse"] * out["ly_reverse"].to_numpy()
            + COMPOSITE_W["ly_ball"] * out["ly_ball"].to_numpy()
            + COMPOSITE_W["ly_middle"] * out["ly_middle"].to_numpy())
    out["ability_composite"] = comp
    return out


def build_global_rates(df):
    n = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    tot = max(n.sum(), 1.0)
    return {
        "success": float(df[TARGET].mean()),
        "reverse": float((df["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n).sum() / tot),
        "ball": float((df["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n).sum() / tot),
        "middle": float((df["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n).sum() / tot),
    }


def export_stats(ly_table, global_rates, seasons_range, k=K_LY):
    return {"ly_table": ly_table, "global_rates": global_rates,
            "seasons_range": list(seasons_range), "k": float(k),
            "composite_w": dict(COMPOSITE_W)}
