"""idea45 — 트랙맨 직접피처(2군활동+시즌간 물리트렌드)를 모델이 직접 읽게 했을 때.

앞선 판정(partial_gain 0.30시그마)은 **선형** 잔차상관이라 두 가지를 못 본다:
  (1) 트리의 비선형/상호작용 활용
  (2) 프록시(tm_matched, Spearman 0.718) 대비 '깨끗한 값'의 품질 이득
그래서 실제로 피처를 넣고 재학습해 비교한다.

측정설계: 페어링(같은 시드끼리 비교)으로 공통 노이즈를 상쇄.
screen_v3에서 확인된 대로 페어SD가 단순 비교보다 훨씬 작다.
CatBoost는 결측을 native 처리하므로 커버리지 67.5%를 그대로 사용.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

CD = "idea45_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7, 2024]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("로드...")
X = pd.read_parquet("featcache_X.parquet")
F = pd.read_parquet("tm_direct_feats.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
mo = pd.read_csv("../data/train.csv", encoding="utf-8-sig", usecols=["game_month"])["game_month"].to_numpy()
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
X2 = pd.concat([X, F], axis=1)
log(f"  기본 {X.shape[1]}피처 -> 확장 {X2.shape[1]}피처 (+{F.shape[1]})")

upto, val = 2023, 2024
tr_m = seasons <= upto
va_m = seasons == val
yv = y[va_m]
mv = mo[va_m]
seg = (mv >= 3) & (mv <= 7)
w = 0.5 ** ((upto - seasons[tr_m]) / 2.0)

CAT = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")


def sc(p, m_):
    yy = yv[m_]
    r = yy.mean(); BS = r * (1 - r)
    return 1e5 * (1 - np.mean((np.clip(p[m_], 0, 1) - yy) ** 2) / BS)


res = {}
for name, Xu in [("base162", X), ("tmdirect173", X2)]:
    preds = []
    for s in SEEDS:
        f = f"{CD}/{name}_s{s}.npy"
        if os.path.exists(f):
            preds.append(np.load(f)); continue
        ts = time.time()
        n_es = int(tr_m.sum() * 0.92)
        m = CatBoostClassifier(**CAT, random_seed=s)
        m.fit(Xu.loc[tr_m].iloc[:n_es], y[tr_m][:n_es], sample_weight=w[:n_es],
              eval_set=(Xu.loc[tr_m].iloc[n_es:], y[tr_m][n_es:]))
        p = m.predict_proba(Xu.loc[va_m])[:, 1]
        np.save(f, p); preds.append(p)
        log(f"    [{name}/s{s}] best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) "
            f"3-7월={sc(p, seg):.2f}")
    res[name] = preds
    avg = np.mean(preds, axis=0)
    log(f"  {name:<12} 3-7월={sc(avg, seg):8.2f}  전체={sc(avg, np.ones(len(mv), bool)):8.2f}")

print()
print("=" * 78)
print("트랙맨 직접피처 효과 (페어링 비교: 같은 시드끼리 차이)")
print("=" * 78)
ALL = np.ones(len(mv), bool)
for segnm, sm in [("3-7월(주판정)", seg), ("전체2024", ALL)]:
    a = [sc(p, sm) for p in res["base162"]]
    b = [sc(p, sm) for p in res["tmdirect173"]]
    pair = [x - y_ for x, y_ in zip(b, a)]
    print(f"{segnm:<14} base={np.mean(a):8.2f}  +tm={np.mean(b):8.2f}  "
          f"Δ={np.mean(b)-np.mean(a):+7.2f}")
    print(f"{'':14} 시드별 페어Δ = {[f'{v:+.2f}' for v in pair]}  "
          f"평균={np.mean(pair):+.2f} SD={np.std(pair):.2f}")
print()
print("판정: 페어Δ 평균이 페어SD보다 크고 부호가 3시드 일치해야 신호")
log(f"총 {time.time()-t0:.0f}s")
