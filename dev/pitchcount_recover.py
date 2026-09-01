"""prev1/prev5_game_* rate에서 실제 투구 수(분모)를 복원한다.

leakage 없음: 이 값들은 organizer가 제공한 그 행 자신의 point-in-time 피처일 뿐 —
다른 행(test 포함) 정보를 전혀 사용하지 않는다. 그 안에 숨어있던 정수 분자/분모를
유리수 근사로 꺼내는 것.

핵심: success_rate 하나만 쓰면 약분 때문에 심하게 과소추정된다 (12/24=0.5 -> n=2).
success_rate와 middle_rate는 같은 등판의 같은 투구 수를 분모로 쓰므로, 둘 다 동시에
정수 분자를 만드는 최소 n을 찾으면 약분 케이스가 대부분 해소된다.

성능: rate 조합의 unique 값에 대해서만 풀고 원본 행에 map back 한다.
"""
import numpy as np
import pandas as pd


def _solve_unique(rate_mat, max_n, tol=1e-6, chunk=20_000):
    """rate_mat: (m, k) — 같은 분모를 공유하는 k개의 rate. 각 행마다 최소 n을 반환."""
    m = len(rate_mat)
    out = np.full(m, np.nan)
    cand = np.arange(1, max_n + 1, dtype=np.float64)
    for s in range(0, m, chunk):
        r = rate_mat[s:s + chunk]                      # (c, k)
        prod = r[:, None, :] * cand[None, :, None]     # (c, n, k)
        resid = np.abs(prod - np.round(prod)).max(axis=2)  # 모든 rate가 동시에 정수여야 함
        ok = resid < tol
        has = ok.any(axis=1)
        best = np.where(ok, cand[None, :], np.inf).argmin(axis=1)
        fb = resid.argmin(axis=1)
        out[s:s + chunk] = cand[np.where(has, best, fb)]
    return out


def recover_denominator(df, rate_cols, max_n, tol=1e-6):
    """rate_cols(같은 분모 공유)의 unique 조합에 대해 분모를 복원해 행 단위로 되돌린다."""
    sub = df[rate_cols]
    valid = sub.notna().all(axis=1).to_numpy()
    n_out = np.full(len(df), np.nan)
    if not valid.any():
        return n_out

    vals = sub[valid].to_numpy(np.float64)
    uniq, inv = np.unique(np.round(vals, 12), axis=0, return_inverse=True)
    solved = _solve_unique(uniq, max_n, tol=tol)
    n_out[valid] = solved[inv]
    return n_out


def build_workload_features(df):
    q1 = recover_denominator(
        df, ["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev1_game_middle_rate"], max_n=200)
    q5 = recover_denominator(
        df, ["asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev5_game_middle_rate"], max_n=600)

    out = pd.DataFrame(index=df.index)
    out["recent1_pitch_n"] = np.log1p(np.nan_to_num(q1, nan=0.0))
    out["recent5_pitch_n"] = np.log1p(np.nan_to_num(q5, nan=0.0))
    avg5 = np.where(~np.isnan(q5), q5 / 5.0, np.nan)
    ratio = np.where((~np.isnan(q1)) & (~np.isnan(avg5)), q1 / (avg5 + 1.0), np.nan)
    out["recent1_vs_avg5_ratio"] = np.nan_to_num(ratio, nan=1.0)  # 결측=평소와 같다고 간주
    out["recent_workload_missing"] = np.isnan(q1).astype(np.float64)
    return out
