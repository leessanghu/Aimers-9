"""아이디어 9 — 반복구간 스냅샷 평균 (거의 공짜 분산감소).

CatBoost는 predict(ntree_end=k)로 학습 도중 임의 시점의 예측을 뽑을 수 있다.
한 번 학습해서 iteration 50%/70%/85%/100% 시점 예측을 평균내면, 추가 학습 비용 0으로
분산이 줄어든다.

*** v38/v39 교훈: 반드시 시드 반복으로 노이즈 바닥 먼저 측정. fold B는 참고만. ***
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from phase2_common import time_split_es

CD = "idea9_cache"
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

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)

CAT_BASE = dict(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")


def fit_snapshots(tag, seed, tr_i, es_i, Xtr, ytr, wtr, Xva):
    f = f"{CD}/{tag}_s{seed}.npz"
    if os.path.exists(f):
        d = np.load(f)
        return d["full"], d["snap_avg"], d["best_iter"].item()
    p = dict(CAT_BASE); p["random_seed"] = seed
    ts = time.time()
    cb = CatBoostClassifier(**p)
    cb.fit(Xtr.iloc[tr_i], ytr[tr_i], sample_weight=wtr[tr_i], eval_set=(Xtr.iloc[es_i], ytr[es_i]))
    best = cb.best_iteration_
    full = cb.predict_proba(Xva)[:, 1]
    snap_iters = sorted(set(max(int(best * f), 10) for f in [0.5, 0.7, 0.85, 1.0]))
    snaps = [cb.predict_proba(Xva, ntree_end=k)[:, 1] for k in snap_iters]
    snap_avg = np.mean(snaps, axis=0)
    np.savez(f, full=full, snap_avg=snap_avg, best_iter=best)
    log(f"    s{seed} best_iter={best} snap_iters={snap_iters} ({time.time()-ts:.0f}s)")
    return full, snap_avg, best


results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = (seasons <= upto) & step
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def sc(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    Xtr = X.loc[tr_m].reset_index(drop=True)
    ytr = y[tr_m]; wtr = w[tr_m]
    Xva = X.loc[va_m]
    tr_i, es_i = time_split_es(len(Xtr))

    fulls, snaps = [], []
    for s in SEEDS:
        full, snap_avg, best = fit_snapshots(tag, s, tr_i, es_i, Xtr, ytr, wtr, Xva)
        fulls.append(full); snaps.append(snap_avg)
        log(f"  s{s}: full={sc(full):.2f}  snap평균(단일시드내)={sc(snap_avg):.2f}  "
            f"corr(full,snap)={np.corrcoef(full,snap_avg)[0,1]:.4f}")

    full_scores = [sc(p) for p in fulls]
    full_avg2seed = np.mean(fulls, axis=0)
    snap_avg2seed = np.mean(snaps, axis=0)
    seed_spread = max(full_scores) - min(full_scores)
    log(f"  [기준선] full 시드별={[round(x,2) for x in full_scores]}  시드폭={seed_spread:.2f}  "
        f"2시드평균={sc(full_avg2seed):.2f}")
    log(f"  [스냅샷] 2시드x스냅 평균={sc(snap_avg2seed):.2f}  "
        f"(2시드full평균 대비 {sc(snap_avg2seed)-sc(full_avg2seed):+.2f})")

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    v35l = 0.55 * base3 + 0.45 * hur
    row = dict(seed_spread=seed_spread, full2=sc(full_avg2seed), snap2=sc(snap_avg2seed), v35local=sc(v35l))
    # single-seed snapshot(비용0) 단독도 기록 -> 실전에서 시드1개로도 되는지
    row["single_seed_gain"] = sc(snaps[0]) - full_scores[0]
    results[tag] = row
    log(f"  [단일시드 내 스냅효과] full={full_scores[0]:.2f}  snap={sc(snaps[0]):.2f}  "
        f"({row['single_seed_gain']:+.2f})  <- 이게 진짜 '공짜' 효과")

print()
print("=" * 78)
print(f"{'fold':<6}{'시드폭':>9}{'full(2s)':>10}{'snap(2s)':>10}{'단일시드내효과':>14}{'v35local':>10}")
for tag, r in results.items():
    print(f"{tag:<6}{r['seed_spread']:9.2f}{r['full2']:10.2f}{r['snap2']:10.2f}"
          f"{r['single_seed_gain']:14.2f}{r['v35local']:10.2f}")
print()
gains_clean = [results[t]["single_seed_gain"] for t in ["A", "C"]]
spreads_clean = [results[t]["seed_spread"] for t in ["A", "C"]]
print(f"클린폴드(A,C) 단일시드내 효과: {[round(g,2) for g in gains_clean]}  "
      f"시드폭: {[round(s,2) for s in spreads_clean]}")
print("=> 노이즈 초과, 유효" if min(gains_clean) > max(spreads_clean) else "=> 노이즈 이내, 판정불가")
pd.DataFrame(results).T.to_csv("idea9_results.csv")
log(f"총 {time.time()-t0:.0f}s")
