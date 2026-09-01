"""아이디어I — 상위 단조(monotonic) 피처 15개에 monotonic_cst 제약을 건 HGB와,
제약 없는 기존(base_d6)을 비교. 편향은 거의 안 늘리고 분산만 줄이는 정규화라는 가설.
fold A 학습데이터로만 방향을 판정(미래정보 안 씀, 전 폴드 공통 사용).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea21_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]

MONO_FEATS = {
    "asof_pitcher_success_rate_smooth": 1, "asof_pitcher_reverse_rate_smooth": -1,
    "asof_batter_n": -1, "asof_batter_success_rate_smooth": 1,
    "asof_pitcher_prev3_game_success_rate": 1, "inseason_success_smooth": 1,
    "inseason_reverse_smooth": -1, "x_ability_here": 1, "x_rev_over_succ": -1,
    "x_mid_over_succ": -1, "bat_inseason_smooth": 1, "bat_inseason_n": -1,
    "inseason_middle_smooth": -1, "inseason_cmd_index": 1, "bat_inseason_middle": -1,
}


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

cols = list(X.columns)
mono_cst = np.zeros(len(cols), dtype=np.int8)
found = 0
for i, c in enumerate(cols):
    if c in MONO_FEATS:
        mono_cst[i] = MONO_FEATS[c]
        found += 1
log(f"단조제약 적용 피처 {found}/{len(MONO_FEATS)}개 매칭 (전체 {len(cols)}피처 중)")

BASE_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
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

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n_}.npy") for n_ in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n_}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n_}.npy")
                   for n_ in ["d6", "d8"]], axis=0)
    v35l = 0.55 * base3 + 0.45 * hur
    d6_orig = np.load(f"phase90_cache/{tag}_base_d6.npy")
    log(f"  v35local={sc(v35l):.2f}  기존d6(무제약)={sc(d6_orig):.2f}")

    p_seeds = []
    for seed in SEEDS:
        f_pred = f"{CD}/{tag}_mono_s{seed}.npy"
        if os.path.exists(f_pred):
            p = np.load(f_pred)
        else:
            ts = time.time()
            m = HistGradientBoostingClassifier(**BASE_PARAMS, monotonic_cst=mono_cst, random_state=seed)
            m.fit(X.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
            p = m.predict_proba(X.loc[va_m])[:, 1]
            np.save(f_pred, p)
            log(f"    s{seed} 학습완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)  단독={sc(p):.2f}")
        p_seeds.append(p)

    p_avg = np.mean(p_seeds, axis=0)
    spread = max(sc(p) for p in p_seeds) - min(sc(p) for p in p_seeds)
    results[tag] = dict(v35local=sc(v35l), d6_orig=sc(d6_orig), mono_avg=sc(p_avg), spread=spread)
    log(f"  d6(무제약)={sc(d6_orig):.2f}  d6(단조제약) 2시드평균={sc(p_avg):.2f}  "
       f"(차이 {sc(p_avg)-sc(d6_orig):+.2f})  시드폭={spread:.2f}")

    base3_swapped = np.mean([p_avg, np.load(f"phase90_cache/{tag}_base_d8.npy"),
                             np.load(f"phase90_cache/{tag}_base_sub.npy")], axis=0)
    v35l_swapped = 0.55 * base3_swapped + 0.45 * hur
    results[tag]["v35l_swapped"] = sc(v35l_swapped)
    log(f"  base3에서 d6만 단조제약판으로 교체 -> v35local={sc(v35l_swapped):.2f}  (원래대비 {sc(v35l_swapped)-sc(v35l):+.2f})")

print()
print("=" * 100)
print(f"{'fold':<6}{'d6무제약':>10}{'d6단조':>10}{'차이':>8}{'시드폭':>8}{'v35l원본':>10}{'v35l(교체)':>12}{'교체이득':>8}")
for tag, r in results.items():
    diff = r["mono_avg"] - r["d6_orig"]
    swap_gain = r["v35l_swapped"] - r["v35local"]
    print(f"{tag:<6}{r['d6_orig']:10.2f}{r['mono_avg']:10.2f}{diff:+8.2f}"
         f"{r['spread']:8.2f}{r['v35local']:10.2f}{r['v35l_swapped']:12.2f}{swap_gain:+8.2f}")
log(f"총 {time.time()-t0:.0f}s")
