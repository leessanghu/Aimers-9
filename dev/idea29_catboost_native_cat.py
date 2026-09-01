"""CatBoost native categorical pitcher_id/batter_id — fold A 주검증.
기존 162피처 + pitcher_id/batter_id를 cat_features로 직접 추가.
cat_d6와 동일 하이퍼파라미터(depth=6, seed=42)로 순수 "native cat 추가효과"만 비교.
새 검증기준: fold A=주검증(방향+크기), fold B=참고(안 망가지는지만), fold C=시드폭참고.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

CD = "idea29_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
CAT_PARAMS_BASE = dict(iterations=3000, learning_rate=0.03, l2_leaf_reg=5.0, verbose=0,
                       early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss", depth=6)


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

X166 = X.copy()
X166["cat_pitcher_id"] = meta["pitcher_id"].astype(str)
X166["cat_batter_id"] = meta["batter_id"].astype(str)
CAT_COLS = ["cat_pitcher_id", "cat_batter_id"]

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2022, 2023, "B"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def sc(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    v35l = 0.55 * base3 + 0.45 * hur
    cat_orig = np.load(f"phase90_cache/{tag}_base_catd6.npy") if os.path.exists(f"phase90_cache/{tag}_base_catd6.npy") \
        else None
    log(f"  v35local={sc(v35l):.2f}")

    p_seeds = []
    for seed in SEEDS:
        f_out = f"{CD}/{tag}_catnative_s{seed}.npy"
        if os.path.exists(f_out):
            p = np.load(f_out)
        else:
            ts = time.time()
            n = tr_m.sum()
            cut = int(n * 0.92)
            Xtr = X166.loc[tr_m]
            idx_tr, idx_es = np.arange(cut), np.arange(cut, n)
            params = dict(CAT_PARAMS_BASE, random_seed=seed)
            m = CatBoostClassifier(**params)
            m.fit(Xtr.iloc[idx_tr], y[tr_m][idx_tr], sample_weight=w[tr_m][idx_tr],
                 eval_set=(Xtr.iloc[idx_es], y[tr_m][idx_es]), cat_features=CAT_COLS)
            p = m.predict_proba(X166.loc[va_m])[:, 1]
            np.save(f_out, p)
            log(f"    s{seed} 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)  단독={sc(p):.2f}")
        p_seeds.append(p)

    p_avg = np.mean(p_seeds, axis=0)
    spread = max(sc(p) for p in p_seeds) - min(sc(p) for p in p_seeds)
    log(f"  native_cat 2시드평균 단독={sc(p_avg):.2f}  시드폭={spread:.2f}")

    # base3에 4번째 멤버로 추가(균등평균 재조정: base4 = mean(d6,d8,sub,native))
    base4 = np.mean([base3, p_avg], axis=0)  # base3는 이미 3개 평균이므로 실질 0.75:0.25 비중
    v35l_added = 0.55 * base4 + 0.45 * hur
    results[tag] = dict(v35local=sc(v35l), added=sc(v35l_added), spread=spread, solo=sc(p_avg))
    log(f"  base3에 native_cat 추가(4번째 멤버) -> v35local={sc(v35l_added):.2f}  "
       f"(원래대비 {sc(v35l_added)-sc(v35l):+.2f})")

print()
print("=" * 90)
print(f"{'fold':<6}{'v35local':>10}{'단독':>10}{'추가후':>10}{'이득':>8}{'시드폭':>8}")
for tag, r in results.items():
    print(f"{tag:<6}{r['v35local']:10.2f}{r['solo']:10.2f}{r['added']:10.2f}"
         f"{r['added']-r['v35local']:+8.2f}{r['spread']:8.2f}")
gain_a = results["A"]["added"] - results["A"]["v35local"]
gain_b = results["B"]["added"] - results["B"]["v35local"]
gain_c = results["C"]["added"] - results["C"]["v35local"]
print(f"\n[신기준] 주검증 fold A={gain_a:+.2f}  {'양수->통과후보' if gain_a>0 else '음수->기각'}")
print(f"참고: fold B(안망가지는지만)={gain_b:+.2f}  fold C(시드폭참고)={gain_c:+.2f} 시드폭={results['C']['spread']:.2f}")
log(f"총 {time.time()-t0:.0f}s")
