"""신호가 실제로 어디 있는지 실측 — 그룹핑별 오라클 천장 (잡음보정).

지금까지는 '이런 피처가 좋을 것 같다'로 후보를 만들고 검증했는데, phase64b에서 폼/trackman/역할이
전부 합쳐 +44점뿐인 게 확인됐다. 추측을 멈추고 신호의 위치를 직접 측정한다.

방법:
  어떤 그룹핑 G(예: pitcher_id, count_state, pitcher x count)에 대해 각 셀의 실제 성공률을
  구하면, 완벽한 모델이 그 그룹핑만으로 달성 가능한 최대 BSS는 Var(E[y|G])/Var(y) 다.
  단 셀 평균은 표본잡음을 포함하므로 반드시 빼야 한다:
      Var(관측 셀평균) = Var(진짜) + E[p(1-p)/n]
      -> Var(진짜) = Var(관측) - E[p(1-p)/n]

이건 '오라클' 천장이다 — 그 시즌의 실현값을 이미 알고 있다고 가정하므로 실제로는 도달 불가능하다.
하지만 상한이므로: 천장이 낮으면 그 방향엔 애초에 신호가 없다는 뜻이고, 천장이 높으면
'추정만 잘하면' 얻을 수 있는 여지가 있다는 뜻이다.

비교 기준: 현재 GBDT 잠재력 = 895.9 (2024 폴드)
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

t0 = time.time()
VALID_SEASON = 2024


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["count_state"] = df["balls_before"] * 4 + df["strikes_before"]
df["same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(int)
df["inn_b"] = np.clip(df["inning"], 1, 9)

d = df[df["season"] == VALID_SEASON].copy()
y = d["control_success"].to_numpy(np.float64)
ybar = y.mean()
bsref = ybar * (1 - ybar)
log(f"2024 n={len(d):,}  base rate={ybar:.4f}  BSref={bsref:.5f}")


def ceiling(keys, min_n=1):
    g = d.groupby(keys)["control_success"]
    n = g.size().to_numpy(np.float64)
    m = g.mean().to_numpy(np.float64)
    ok = n >= min_n
    n, m = n[ok], m[ok]
    w = n / n.sum()
    var_obs = float(np.sum(w * (m - ybar) ** 2))
    noise = float(np.sum(w * m * (1 - m) / np.maximum(n, 1)))
    var_true = max(0.0, var_obs - noise)
    return {
        "cells": int(len(n)),
        "median_n": float(np.median(n)),
        "score_raw": 1e5 * var_obs / bsref,
        "score_true": 1e5 * var_true / bsref,
        "sd_true": float(np.sqrt(var_true)),
    }


GROUPS = [
    ("count_state (볼/스트 카운트)", ["count_state"]),
    ("inning", ["inn_b"]),
    ("game_type", ["game_type"]),
    ("base_state", ["base_state"]),
    ("game_month", ["game_month"]),
    ("pitcher_id", ["pitcher_id"]),
    ("batter_id", ["batter_id"]),
    ("pitcher_team_id", ["pitcher_team_id"]),
    ("batter_team_id", ["batter_team_id"]),
    ("--- 2원 조합 ---", None),
    ("pitcher x count", ["pitcher_id", "count_state"]),
    ("pitcher x inning", ["pitcher_id", "inn_b"]),
    ("pitcher x batter_hand", ["pitcher_id", "batter_hand"]),
    ("pitcher x base_state", ["pitcher_id", "base_state"]),
    ("batter x count", ["batter_id", "count_state"]),
    ("count x inning", ["count_state", "inn_b"]),
    ("count x base_state", ["count_state", "base_state"]),
    ("pitcher x batter (h2h)", ["pitcher_id", "batter_id"]),
    ("--- 상황 전체 ---", None),
    ("count x base x inning x outs", ["count_state", "base_state", "inn_b", "outs_before"]),
    ("pitcher x count x hand", ["pitcher_id", "count_state", "batter_hand"]),
]

log("\n" + "=" * 92)
log("그룹핑별 오라클 천장 (2024 실현값을 다 안다고 가정한 상한)")
log("=" * 92)
print(f"{'그룹핑':<34}{'셀수':>9}{'중앙n':>8}{'보정전':>10}{'진짜천장':>11}{'SD':>9}")
print("-" * 82)
for name, keys in GROUPS:
    if keys is None:
        print(f"{name}")
        continue
    c = ceiling(keys)
    print(f"{name:<34}{c['cells']:9,}{c['median_n']:8.0f}{c['score_raw']:10.0f}"
          f"{c['score_true']:11.0f}{c['sd_true']:9.4f}")

print()
print("참고:")
print(f"  현재 GBDT 잠재력 (2024)     895.9   (pred_std 0.0466)")
print(f"  LB 1500 로컬환산            1316.1")
print()
print("읽는 법: '진짜천장'은 그 그룹핑만으로 완벽히 예측했을 때의 상한이다.")
print("  전체 대비 낮으면 그 방향엔 신호가 없다. 높으면 '추정을 잘하면' 여지가 있다.")
print("  단 오라클이므로 실제 달성은 항상 이보다 낮다 (미래 시즌은 알 수 없으므로).")
