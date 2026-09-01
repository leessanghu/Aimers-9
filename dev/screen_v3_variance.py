"""스크리닝 v3 — 측정 장비 자체를 고친다.

문제 진단 (idea40/idea42에서 확정):
  stage3 7/7 전부 |Δ| < 시드폭. 즉 **단 한 건도 노이즈를 넘지 못했다.**
    streak -8.19(폭42.65) / workload -5.66(폭6.80) / hot_only -4.57(폭29.00)
    marcel_full -10.76(폭31.18) / marcel_dev_only -12.15(폭14.68) / marcel_multi +3.17(폭51.05)
  그런데 우리가 찾는 효과크기는 fold A 블렌드 기준 ~1점이다
    (midaxis fold A +1.08 -> 실측 +7.72 / unified5 +0.78 -> +6.99).
  신호:잡음 = 1 : 15~50. **원리적으로 탐지 불가능한 장비로 7번 측정한 것.**
  => "신호가 없다"가 아니라 "볼 수 없었다"가 정확한 결론.

분산 원인 후보:
  (1) early_stopping=True, validation_fraction=0.1 -> ES용 랜덤 10% 분할이
      시드마다 달라 최적 iteration이 흔들림(관측: iters 416~500). 지배적 의심.
  (2) HGB 내부 binning 서브샘플
  (3) 단일 모델 vs 앙상블 (세션 기록: d6단독 20.27 -> base3평균 6.57)

이 스크립트는 (1)을 고정(early_stopping=False, max_iter 고정)했을 때
시드폭이 얼마나 줄어드는지 측정하고, 동일 조건에서 후보 델타가
탐지 가능해지는지 확인한다. 후보는 marcel_dev_only(단일피처, 기존 폭 14.68로 가장 작았음).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "screen_v3_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7, 2024, 1234, 99]  # 5시드로 확대


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
raw = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                  usecols=["season", "game_month", "pitcher_id", "control_success"])
y = raw["control_success"].to_numpy(np.float64)
seasons = raw["season"].to_numpy(np.float64)
mo = raw["game_month"].to_numpy()
g_rate = float(y.mean())
sr = sorted(np.unique(seasons).tolist())

# marcel_dev (idea41/42에서 stage1 3.95σ, Spearman 0.517로 가장 새로웠던 후보)
ys = raw.groupby(["pitcher_id", "season"])["control_success"].agg(s="sum", n="count")
S_ = np.nan_to_num(ys["s"].unstack().reindex(columns=sr).to_numpy())
N_ = np.nan_to_num(ys["n"].unstack().reindex(columns=sr).to_numpy())
mS = np.zeros_like(S_); mN = np.zeros_like(N_)
for j in range(len(sr)):
    for lag, wt in [(1, 5.0), (2, 4.0), (3, 3.0)]:
        if j - lag >= 0:
            mS[:, j] += wt * S_[:, j - lag]; mN[:, j] += wt * N_[:, j - lag]
marcel = (mS + 600.0 * g_rate) / (mN + 600.0)
mt = pd.DataFrame(marcel, index=ys["s"].unstack().index, columns=sr).stack(future_stack=True)
idxm = pd.MultiIndex.from_arrays([raw["pitcher_id"], raw["season"]])
mval = pd.Series(mt.reindex(idxm).to_numpy()).fillna(g_rate).to_numpy(np.float64)
MARCEL_DEV = mval - X["asof_pitcher_success_rate_smooth"].to_numpy()

tr_m = seasons <= 2023
va = seasons == 2024
yv = y[va]; mv = mo[va]
seg = (mv >= 3) & (mv <= 7)
w = 0.5 ** ((2023 - seasons[tr_m]) / 2.0)


def sc(p, m_):
    yy = yv[m_]; r = yy.mean(); BS = r * (1 - r)
    return 1e5 * (1 - np.mean((np.clip(p[m_], 0, 1) - yy) ** 2) / BS)


ARMS = [
    # (이름, ES사용여부, max_iter, 후보포함여부)
    ("ES_base", True, 500, False),
    ("ES_marcel", True, 500, True),
    ("FIX_base", False, 400, False),
    ("FIX_marcel", False, 400, True),
]
COMMON = dict(learning_rate=0.03, l2_regularization=5.0, max_depth=6, max_leaf_nodes=31)

res = {}
for name, use_es, mi, add in ARMS:
    preds = []
    for s in SEEDS:
        f = f"{CD}/{name}_s{s}.npy"
        if os.path.exists(f):
            preds.append(np.load(f)); continue
        ts = time.time()
        Xu = X if not add else pd.concat([X, pd.Series(MARCEL_DEV, index=X.index, name="marcel_dev")], axis=1)
        kw = dict(COMMON, max_iter=mi, random_state=s)
        if use_es:
            kw.update(early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)
        else:
            kw.update(early_stopping=False)
        m_ = HistGradientBoostingClassifier(**kw)
        m_.fit(Xu.loc[tr_m], y[tr_m], sample_weight=w)
        p = m_.predict_proba(Xu.loc[va])[:, 1]
        np.save(f, p); preds.append(p)
        log(f"    [{name}/s{s}] iters={m_.n_iter_} ({time.time()-ts:.0f}s)")
    s37 = [sc(p, seg) for p in preds]
    savg = sc(np.mean(preds, axis=0), seg)
    res[name] = dict(avg=savg, spread=max(s37) - min(s37), sd=float(np.std(s37)), each=s37)
    log(f"  {name:<11} 3-7월 평균={savg:8.2f}  시드폭={max(s37)-min(s37):6.2f}  시드SD={np.std(s37):6.2f}")

print()
print("=" * 84)
print("측정 장비 비교 — early stopping 랜덤분할 제거가 분산을 줄이는가")
print("=" * 84)
print(f"{'조건':<14}{'3-7월평균':>11}{'시드폭':>9}{'시드SD':>9}")
for k, v in res.items():
    print(f"{k:<14}{v['avg']:11.2f}{v['spread']:9.2f}{v['sd']:9.2f}")
print()
for tag, bn, mn in [("ES(기존)", "ES_base", "ES_marcel"), ("FIX(고정iter)", "FIX_base", "FIX_marcel")]:
    if bn in res and mn in res:
        d = res[mn]["avg"] - res[bn]["avg"]
        # 페어링된 시드별 델타 (같은 시드끼리 비교하면 공통 노이즈가 상쇄됨)
        pair = [a - b for a, b in zip(res[mn]["each"], res[bn]["each"])]
        print(f"{tag:<14} marcel_dev Δ={d:+7.2f}   시드별페어Δ={[f'{x:+.1f}' for x in pair]}  "
              f"페어평균={np.mean(pair):+.2f} 페어SD={np.std(pair):.2f}")
print()
print("판정기준: 페어SD가 충분히 작아야(<3) fold A ~1점 효과를 탐지할 수 있다.")
log(f"총 {time.time()-t0:.0f}s")
