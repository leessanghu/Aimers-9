"""3단계 — D_count_spin_std(season,count_state별 spin_rate 표준편차) 3폴드+시드반복.
2단계 split-half 통과(+10.83±2.58, 판정기준 +9.10) 유일 생존자.
leakage 안전: 직전시즌(season-1)의 트릭맨 집계값을 조회 (동일시즌 사용 안 함).
target-free라 리그 전체 통계, 개인 아님 -> 셀 크기 매우 큼(문제 없음).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea28_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

log("트릭맨 로드 + (season,count_state)별 spin_rate std 집계...")
tm = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig",
                 usecols=["season", "balls_before", "strikes_before", "spin_rate"])
tm["count_state"] = tm["balls_before"] * 4 + tm["strikes_before"]
agg = tm.groupby(["season", "count_state"])["spin_rate"].std().reset_index()
agg.columns = ["season", "count_state", "spin_std"]
agg["season"] = agg["season"] + 1  # 다음 시즌 행에서 조회하도록 시프트 (직전시즌 값)
global_fallback = float(tm["spin_rate"].std())

count_state = meta["balls_before"].to_numpy() * 4 + meta["strikes_before"].to_numpy()
lookup_df = pd.DataFrame({"season": seasons, "count_state": count_state})
merged = lookup_df.merge(agg, on=["season", "count_state"], how="left")
spin_std_feat = merged["spin_std"].fillna(global_fallback).to_numpy(np.float64)
log(f"  결측률(직전시즌 없음, 2019행)={merged['spin_std'].isna().mean()*100:.2f}%  "
   f"std범위=[{spin_std_feat.min():.1f}, {spin_std_feat.max():.1f}]")

HGB_CLS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
              early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
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
    d6_orig = np.load(f"phase90_cache/{tag}_base_d6.npy")
    log(f"  v35local={sc(v35l):.2f}  기존d6={sc(d6_orig):.2f}")

    X163 = X.copy()
    X163["tm_count_spin_std"] = spin_std_feat

    p_seeds = []
    for seed in SEEDS:
        f_out = f"{CD}/{tag}_d6plusTMspin_s{seed}.npy"
        if os.path.exists(f_out):
            p = np.load(f_out)
        else:
            ts = time.time()
            m = HistGradientBoostingClassifier(**HGB_CLS, random_state=seed)
            m.fit(X163.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
            p = m.predict_proba(X163.loc[va_m])[:, 1]
            np.save(f_out, p)
            log(f"    s{seed} 학습완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)  단독={sc(p):.2f}")
        p_seeds.append(p)

    p_avg = np.mean(p_seeds, axis=0)
    spread = max(sc(p) for p in p_seeds) - min(sc(p) for p in p_seeds)
    log(f"  d6+tmspin(163피처) 2시드평균 단독={sc(p_avg):.2f}  (기존d6대비 {sc(p_avg)-sc(d6_orig):+.2f})  시드폭={spread:.2f}")

    base3_swapped = np.mean([p_avg, np.load(f"phase90_cache/{tag}_base_d8.npy"),
                             np.load(f"phase90_cache/{tag}_base_sub.npy")], axis=0)
    v35l_swapped = 0.55 * base3_swapped + 0.45 * hur
    results[tag] = dict(v35local=sc(v35l), swapped=sc(v35l_swapped), spread=spread)
    log(f"  base3에서 d6만 163피처판으로 교체 -> v35local={sc(v35l_swapped):.2f}  (원래대비 {sc(v35l_swapped)-sc(v35l):+.2f})")

print()
print("=" * 90)
print(f"{'fold':<6}{'v35local':>10}{'교체후':>10}{'이득':>8}{'시드폭':>8}")
for tag, r in results.items():
    print(f"{tag:<6}{r['v35local']:10.2f}{r['swapped']:10.2f}{r['swapped']-r['v35local']:+8.2f}{r['spread']:8.2f}")
gains_clean = [results[t]["swapped"] - results[t]["v35local"] for t in ["A", "C"]]
gain_b = results["B"]["swapped"] - results["B"]["v35local"]
max_spread = max(results[t]["spread"] for t in ["A", "C"])
print(f"\n클린폴드(A,C) 최소이득={min(gains_clean):+.2f}  시드폭최대={max_spread:.2f}  "
     f"{'신뢰가능' if min(gains_clean) > max_spread else '신뢰불가'}  (참고 B={gain_b:+.2f})")
log(f"총 {time.time()-t0:.0f}s")
