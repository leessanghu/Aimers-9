"""Phase95 block1 — base CatBoost(cats, d6/d8/rsm) refit-closure 검증.
v29 이후 한 번도 refit 안 된 채 마지막 8%(가장 최근 데이터)를 ES 용도로만 쓰고
실제 트리 분할엔 못 쓰게 버려왔음(hgb/hurdle은 v44/v45에서 이미 refit-closure 완료,
cats만 누락). ES로 확정한 iteration을 고정하고 train 전체로 재학습.
fold A/C, featcache 재사용(빠름). 실제 프로덕션 base=0.5*hgb+0.5*cat 구조에 꽂아
전체 v47 구성 기준 delta를 측정한다. ("ES방식 변경류" -> 세션 규칙상 편향 거의 없음)
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

CD90 = "phase90_cache"
CD13 = "idea13_cache"
CD = "idea35_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()

CONFIGS = [
    ("d6", dict(depth=6, random_seed=42)),
    ("d8", dict(depth=8, l2_leaf_reg=10.0, random_seed=7)),
    ("rsm", dict(depth=6, rsm=0.6, random_seed=2024)),
]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def sc(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    Xtr = X.loc[tr_m].reset_index(drop=True)
    ytr = y[tr_m]
    wtr = w[tr_m]
    Xva = X.loc[va_m]
    n_tr = len(Xtr)
    n_es = int(n_tr * 0.92)

    hgb3 = np.mean([np.load(f"{CD90}/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"{CD90}/{tag}_core_{n}.npy")) * np.load(f"{CD90}/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    mr = np.mean([np.load(f"{CD13}/{tag}_multires_s{k}.npy") for k in [42, 7]], axis=0)
    od = np.mean([np.load(f"{CD13}/{tag}_ordinal_s{k}.npy") for k in [42, 7]], axis=0)

    cat_ES, cat_refit = [], []
    for name, extra in CONFIGS:
        f_es = f"{CD}/{tag}_cat_{name}_ES.npy"
        f_rf = f"{CD}/{tag}_cat_{name}_refit.npy"
        if os.path.exists(f_es) and os.path.exists(f_rf):
            p_es = np.load(f_es); p_rf = np.load(f_rf)
        else:
            params = dict(iterations=3000, learning_rate=0.03, l2_leaf_reg=5.0, verbose=0,
                         early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
            params.update(extra)
            ts = time.time()
            m_es = CatBoostClassifier(**params)
            m_es.fit(Xtr.iloc[:n_es], ytr[:n_es], sample_weight=wtr[:n_es],
                    eval_set=(Xtr.iloc[n_es:], ytr[n_es:]))
            best_iter = max(m_es.best_iteration_, 1)
            p_es = m_es.predict_proba(Xva)[:, 1]
            np.save(f_es, p_es)
            log(f"  [{tag}/{name}] ES best_iter={best_iter} score={sc(p_es):.2f} ({time.time()-ts:.0f}s)")

            ts = time.time()
            params_fixed = dict(params); params_fixed.pop("early_stopping_rounds")
            params_fixed["iterations"] = best_iter
            m_rf = CatBoostClassifier(**params_fixed)
            m_rf.fit(Xtr, ytr, sample_weight=wtr)
            p_rf = m_rf.predict_proba(Xva)[:, 1]
            np.save(f_rf, p_rf)
            log(f"  [{tag}/{name}] refit score={sc(p_rf):.2f} ({time.time()-ts:.0f}s)")
        cat_ES.append(p_es); cat_refit.append(p_rf)

    cat3_ES = np.mean(cat_ES, axis=0)
    cat3_rf = np.mean(cat_refit, axis=0)

    base_ES = 0.5 * hgb3 + 0.5 * cat3_ES
    base_rf = 0.5 * hgb3 + 0.5 * cat3_rf
    v47_ES = 0.30 * base_ES + 0.40 * hur + 0.10 * mr + 0.20 * od
    v47_rf = 0.30 * base_rf + 0.40 * hur + 0.10 * mr + 0.20 * od
    results[tag] = dict(es=sc(v47_ES), rf=sc(v47_rf), delta=sc(v47_rf) - sc(v47_ES))
    log(f"  fold {tag}: v47(cat=ES)={sc(v47_ES):.2f}  v47(cat=refit)={sc(v47_rf):.2f}  "
       f"delta={sc(v47_rf)-sc(v47_ES):+.2f}")

print()
print("=" * 60)
for tag, r in results.items():
    print(f"fold {tag}: ES={r['es']:.2f} refit={r['rf']:.2f} delta={r['delta']:+.2f}")
log(f"총 {time.time()-t0:.0f}s")
