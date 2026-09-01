"""투수 실력의 상태공간(칼만) 추정 — asof(커리어누적) + inseason(시즌한정) 두 조각을 하나로 통합.

현재 구조의 문제: 커리어 누적은 옛 시즌을 과대반영하고, in-season은 시즌 초 표본이 없다.
원래 이건 '시간에 따라 변하는 잠재 실력을 관측으로부터 추정'하는 필터링 문제다.

모델:
  상태  theta_s = theta_{s-1} + w,  w ~ N(0, q)      (시즌 간 실력 드리프트)
  관측  p_s ~ N(theta_s, R_s),  R_s = p(1-p)/n_s     (그 시즌 성공률, 이항 노이즈)
  q는 데이터에서 추정: Var(시즌간 성공률 변화) - 이항노이즈 (= 진짜 드리프트 분산)

각 행의 피처:
  1) 직전 시즌까지 필터링 후 1스텝 예측한 실력 theta_pred (+ 그 불확실성 P_pred)
  2) 거기에 '그 행 시점까지의 이번 시즌 관측'을 정밀도 가중으로 갱신한 사후 추정

leakage: theta_pred는 직전 시즌까지의 train 라벨만 사용. 시즌 내 갱신은 그 행 자신의
asof 컬럼(= 그 투구 직전까지)만 사용. 같은 시즌 다른 행/test 다른 행 참조 없음.
"""

import numpy as np
import pandas as pd

TARGET = "control_success"
KALMAN_COLS = ["kal_pred", "kal_prec", "kal_post", "kal_post_minus_pred"]


def estimate_process_noise(df, entity="pitcher_id", min_n=200):
    """시즌 간 실력 변화의 '진짜' 분산 q (이항 노이즈 제거)."""
    g = df.groupby([entity, "season"])[TARGET].agg(s="sum", n="count").reset_index()
    g = g[g["n"] >= min_n]
    g["p"] = g["s"] / g["n"]
    g = g.sort_values([entity, "season"])
    prev = g.groupby(entity).shift(1)
    ok = prev["p"].notna()
    d = (g.loc[ok, "p"] - prev.loc[ok, "p"]).to_numpy()
    noise = (g.loc[ok, "p"] * (1 - g.loc[ok, "p"]) / g.loc[ok, "n"]
             + prev.loc[ok, "p"] * (1 - prev.loc[ok, "p"]) / prev.loc[ok, "n"]).to_numpy()
    return float(max(np.var(d) - np.mean(noise), 1e-6))


def build_kalman_table(df, seasons_range, q, global_rate, entity="pitcher_id"):
    """(entity, season) -> 그 시즌까지 필터링한 뒤 '다음 시즌으로 1스텝 예측'한 (theta, P).

    반환된 행 (e, S) 는 '시즌 S+1 시점에 쓸 수 있는 사전 추정'이다."""
    g = df.groupby([entity, "season"])[TARGET].agg(s="sum", n="count").reset_index()
    piv_s = g.pivot(index=entity, columns="season", values="s").reindex(columns=seasons_range)
    piv_n = g.pivot(index=entity, columns="season", values="n").reindex(columns=seasons_range)
    ents = piv_s.index.to_numpy()

    S = piv_s.to_numpy(np.float64)
    N = piv_n.to_numpy(np.float64)
    m = len(ents)
    # 사전: 전역 평균, 큰 불확실성
    theta = np.full(m, global_rate)
    P = np.full(m, 0.01)
    out_theta = np.full((m, len(seasons_range)), np.nan)
    out_P = np.full((m, len(seasons_range)), np.nan)

    for j in range(len(seasons_range)):
        n_j, s_j = N[:, j], S[:, j]
        seen = np.isfinite(n_j) & (n_j > 0)
        if seen.any():
            p_obs = np.divide(s_j, n_j, out=np.full(m, np.nan), where=seen)
            R = np.divide(p_obs * (1 - p_obs), n_j, out=np.full(m, np.inf), where=seen)
            R = np.maximum(R, 1e-8)
            K = np.where(seen, P / (P + R), 0.0)
            theta = np.where(seen, theta + K * (np.nan_to_num(p_obs) - theta), theta)
            P = np.where(seen, (1 - K) * P, P)
        # 다음 시즌으로 예측 (드리프트 반영)
        theta_next, P_next = theta, P + q
        out_theta[:, j] = theta_next
        out_P[:, j] = P_next
        P = P_next

    t = pd.DataFrame(out_theta, index=ents, columns=seasons_range).stack(future_stack=True)
    p = pd.DataFrame(out_P, index=ents, columns=seasons_range).stack(future_stack=True)
    return t, p


def transform_kalman(df, theta_tbl, P_tbl, global_rate, entity="pitcher_id",
                     n_col="asof_pitcher_n", rate_col="asof_pitcher_success_rate",
                     inseason_n=None, inseason_rate=None):
    """행별 칼만 피처. inseason_n/rate가 주어지면 시즌 내 관측으로 사후 갱신."""
    idx = pd.MultiIndex.from_arrays([df[entity].to_numpy(), df["season"].to_numpy() - 1])
    theta_pred = pd.Series(theta_tbl.reindex(idx).to_numpy()).fillna(global_rate).to_numpy(np.float64)
    P_pred = pd.Series(P_tbl.reindex(idx).to_numpy()).fillna(0.01).to_numpy(np.float64)
    P_pred = np.maximum(P_pred, 1e-8)

    out = pd.DataFrame(index=df.index)
    out["kal_pred"] = theta_pred
    out["kal_prec"] = np.log1p(1.0 / P_pred)

    if inseason_n is None:
        out["kal_post"] = theta_pred
        out["kal_post_minus_pred"] = 0.0
        return out

    n_in = np.asarray(inseason_n, dtype=np.float64)
    p_in = np.nan_to_num(np.asarray(inseason_rate, dtype=np.float64), nan=global_rate)
    var_obs = np.where(n_in > 0, np.maximum(p_in * (1 - p_in), 1e-4) / np.maximum(n_in, 1), np.inf)
    w_pred = 1.0 / P_pred
    w_obs = np.where(np.isfinite(var_obs), 1.0 / var_obs, 0.0)
    post = (theta_pred * w_pred + p_in * w_obs) / (w_pred + w_obs)
    out["kal_post"] = post
    out["kal_post_minus_pred"] = post - theta_pred
    return out


def export_stats(theta_tbl, P_tbl, global_rate, q, seasons_range):
    return {"theta": theta_tbl, "P": P_tbl, "global_rate": float(global_rate),
            "q": float(q), "seasons_range": list(seasons_range)}
