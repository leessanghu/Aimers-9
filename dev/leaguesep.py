"""리그(game_type F/R) 분리 투수 통계 — asof 누적의 리그 혼합 오염을 교정.

발견 (phase 게임타입 조사):
  - game_type F(퓨처스 추정)의 성공률 0.603 vs R 0.514 (전체 +8.9%p)
  - 같은 투수 내에서도 F-R 성공률 차이 평균 +13.3%p, 95.5% 투수에서 F가 높음
  - 792명 중 453명(57%)이 두 리그를 오감
  - 그런데 asof_pitcher_* 누적은 F/R을 섞어서 계산됨
    -> 2군 성적이 1군 예측의 실력 추정치를 체계적으로 오염

피처 (각 행은 자기 행의 game_type + 직전 시즌까지 테이블만 조회, 행 간 참조 없음):
  lg_own_rate   : 그 행과 같은 리그에서의 커리어 성공률 (직전 시즌 끝까지, 스무딩)
  lg_own_n      : 같은 리그 누적 표본 (log1p)
  lg_diff       : own_rate - 혼합 prior  (리그 오염 교정량 = 순수 신규 신호)
  lg_share      : 그 투수 경험 중 자기 리그 비중
"""

import numpy as np
import pandas as pd

LG_COLS = ["lg_own_rate", "lg_own_n", "lg_diff", "lg_share"]
K_LG = 60.0


def build_league_table(df, target_col="control_success"):
    """(pitcher_id, game_type, season) -> 그 시즌 끝까지의 리그별 누적 (s, n). train 라벨로만."""
    g = (df.groupby(["pitcher_id", "game_type", "season"])[target_col]
           .agg(s="sum", n="count").sort_index())
    cum = g.groupby(level=[0, 1]).cumsum()
    return cum.reset_index()


def league_global_rates(df, target_col="control_success"):
    return df.groupby("game_type")[target_col].mean().to_dict()


def transform_league(df, lg_table, lg_globals, mixed_prior_rate, seasons_range, k=K_LG):
    """각 행에 '자기 리그' 기준 직전 시즌까지 통계를 붙인다.

    mixed_prior_rate: 기존 혼합 prior(in-season 모듈의 직전시즌 투수 성공률) — lg_diff 기준점."""
    pid = df["pitcher_id"].to_numpy()
    gt = df["game_type"].astype(str).to_numpy()
    prev = df["season"].to_numpy() - 1

    piv_s = (lg_table.pivot_table(index=["pitcher_id", "game_type"], columns="season",
                                  values="s", aggfunc="first")
             .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True))
    piv_n = (lg_table.pivot_table(index=["pitcher_id", "game_type"], columns="season",
                                  values="n", aggfunc="first")
             .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True))
    # 전체(리그 무관) 누적: share 분모용
    tot = lg_table.groupby(["pitcher_id", "season"])[["n"]].sum().reset_index()
    piv_tot = (tot.pivot_table(index="pitcher_id", columns="season", values="n", aggfunc="first")
               .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True))

    idx_own = pd.MultiIndex.from_arrays([pid, gt, prev])
    s_own = np.nan_to_num(piv_s.reindex(idx_own).to_numpy().astype(np.float64), nan=0.0)
    n_own = np.nan_to_num(piv_n.reindex(idx_own).to_numpy().astype(np.float64), nan=0.0)
    n_tot = np.nan_to_num(piv_tot.reindex(pd.MultiIndex.from_arrays([pid, prev]))
                          .to_numpy().astype(np.float64), nan=0.0)

    gm = pd.Series(gt).map(lg_globals).fillna(np.mean(list(lg_globals.values()))).to_numpy(np.float64)
    own_rate = (s_own + k * gm) / (n_own + k)

    mixed = np.asarray(mixed_prior_rate, dtype=np.float64)
    out = pd.DataFrame(index=df.index)
    out["lg_own_rate"] = own_rate
    out["lg_own_n"] = np.log1p(n_own)
    out["lg_diff"] = own_rate - mixed
    out["lg_share"] = np.divide(n_own, n_tot, out=np.full(len(df), 1.0), where=n_tot > 0)
    return out


def export_stats(lg_table, lg_globals, seasons_range, k=K_LG):
    return {"lg_table": lg_table, "lg_globals": {str(a): float(b) for a, b in lg_globals.items()},
            "seasons_range": list(seasons_range), "k": float(k)}
