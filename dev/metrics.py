"""로컬 평가 지표 — EVALUATION.md의 공식 산식을 그대로 구현.

Score = max(0, 100000 * (1 - Brier Score / 베이스라인 Brier Score))
Brier Score = mean((p_i - y_i)^2)
베이스라인 Brier Score = r * (1 - r), r = mean(y_i)
"""

import numpy as np


def brier_score(y_true, p):
    y_true = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean((p - y_true) ** 2))


def evaluate(y_true, p):
    """Brier Score / Skill Score / 리더보드 환산 점수를 한 번에 반환."""
    y_true = np.asarray(y_true, dtype=np.float64)
    bs = brier_score(y_true, p)
    r = float(y_true.mean())
    baseline_bs = r * (1 - r)
    bss = 1 - bs / baseline_bs if baseline_bs > 0 else float("nan")
    return {
        "n": len(y_true),
        "r": r,
        "brier_score": bs,
        "baseline_brier": baseline_bs,
        "bss": bss,
        "leaderboard_score": max(0.0, 100000 * bss),
    }


def format_report(name, m):
    return (f"{name:28s}  n={m['n']:>8,}  r={m['r']:.4f}  "
            f"BS={m['brier_score']:.6f}  baseBS={m['baseline_brier']:.6f}  "
            f"BSS={m['bss']:.4f}  score={m['leaderboard_score']:.1f}")
