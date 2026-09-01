"""제출권 0회로 테스트 분포 역추론 — 기존 실측 이력을 연립방정식으로 푼다.

## 원리
aux head i를 넣었을 때 실측 델타는, 테스트의 세그먼트 구성비 w로 가중된
로컬 세그먼트별 델타의 함수여야 한다:

    LB_delta_i  ~  a + b * [ w * Δ_i(세그먼트S) + (1-w) * Δ_i(S여집합) ]

미지수 3개(a, b, w), 방정식 5개(실측 아는 aux head 5종) -> 과결정계이므로 풀린다.
w를 격자탐색하며 각 w에서 a,b를 OLS로 적합하고, 잔차가 최소인 w를 채택.

## 검증장치
rookie 축은 **프로브로 이미 답을 알고 있다(7.4~10.4%)**.
같은 방법으로 rookie share를 추정해서 그 범위가 나오면 -> 방법론 신뢰 가능.
그 다음 month37 추정치를 믿을 수 있다(제출권 절약).

실측 5종: midaxis +7.72 / unified5 +6.99 / other +3.25 / ball +1.83 / strike +0.20
(전부 v47 기준 w=0.10 추가, 동일 프로토콜)
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

t0 = time.time()

LB = {"midaxis": 7.72, "unified5": 6.99, "other": 3.25, "ball": 1.83, "strike": 0.20}
SRC = {
    "midaxis": ("idea31_cache", "midaxis", [42, 7]),
    "unified5": ("idea12_cache", "head0", [42, 7]),
    "other": ("idea33_cache", "other", [42, 7]),
    "ball": ("idea32_cache", "ball", [42, 7]),
    "strike": ("idea32_cache", "strike", [42, 7, 2024]),
}

meta = pd.read_parquet("featcache_meta.parquet")
raw = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                  usecols=["game_month", "asof_pitcher_n"])
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
mo = raw["game_month"].to_numpy()
an = raw["asof_pitcher_n"].fillna(0).to_numpy(np.float64)

va = seasons == 2024
yv = y[va]
A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)
base = A([f"phase90_cache/A_base_{n}.npy" for n in ["d6", "d8", "sub"]])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
mr = A([f"idea13_cache/A_multires_s{k}.npy" for k in [42, 7]])
od = A([f"idea13_cache/A_ordinal_s{k}.npy" for k in [42, 7]])
V47 = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od


def seg_score(p, m):
    yy = yv[m]
    r = yy.mean()
    return 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yy) ** 2) / (r * (1 - r)))


# 세그먼트 정의 (fold A 2024 기준)
prev = set(pid[seasons <= 2023])
SEGS = {
    "month37": ((mo[va] >= 3) & (mo[va] <= 7)),
    "rookie": ~pd.Series(pid[va]).isin(prev).to_numpy(),
    "lown500": (an[va] < 500),
}


def solve(segmask, label, truth=None):
    d_in, d_out, lbs = [], [], []
    for name, (d, stem, seeds) in SRC.items():
        c = A([f"{d}/A_{stem}_s{k}.npy" for k in seeds])
        p = 0.90 * V47 + 0.10 * c
        d_in.append(seg_score(p, segmask) - seg_score(V47, segmask))
        d_out.append(seg_score(p, ~segmask) - seg_score(V47, ~segmask))
        lbs.append(LB[name])
    d_in, d_out, lbs = map(np.array, (d_in, d_out, lbs))

    best = (1e18, None, None, None)
    for w in np.arange(0.0, 1.0001, 0.005):
        x = w * d_in + (1 - w) * d_out
        M = np.column_stack([np.ones(len(x)), x])
        coef, *_ = np.linalg.lstsq(M, lbs, rcond=None)
        res = lbs - M @ coef
        rss = float((res ** 2).sum())
        if rss < best[0]:
            best = (rss, w, coef, res)
    rss, w, coef, res = best
    print(f"\n[{label}]  fold A 실제비율 {segmask.mean()*100:.2f}%")
    print(f"  세그먼트별 로컬 델타:")
    for name, a_, b_ in zip(SRC, d_in, d_out):
        print(f"    {name:<10} 내부Δ={a_:+7.2f}  외부Δ={b_:+7.2f}   실측LB={LB[name]:+.2f}")
    print(f"  -> 추정 테스트 비율 w = {w*100:.1f}%   (잔차SD={res.std():.2f}, "
          f"회귀 LB={coef[0]:+.2f}{coef[1]:+.2f}*x)")
    if truth:
        print(f"  ** 프로브 정답: {truth} **")
    # w 프로파일 (얼마나 뾰족한가)
    prof = []
    for w2 in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = w2 * d_in + (1 - w2) * d_out
        M = np.column_stack([np.ones(len(x)), x])
        c2, *_ = np.linalg.lstsq(M, lbs, rcond=None)
        prof.append((w2, float(((lbs - M @ c2) ** 2).sum())))
    print("  잔차제곱합 프로파일: " + "  ".join(f"w={a:.1f}:{b:.1f}" for a, b in prof))
    return w


print("=" * 78)
print("제출권 0회 테스트 분포 역추론 (실측 5건 연립)")
print("=" * 78)
solve(SEGS["rookie"], "rookie (검증용)", truth="7.4~10.4%")
solve(SEGS["month37"], "month37")
solve(SEGS["lown500"], "lown500")
print(f"\n[{time.time()-t0:.0f}s] 완료")
