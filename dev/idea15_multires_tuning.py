"""아이디어F — v40(multires)의 fold C 불안정성(시드폭 93) 원인 진단 + 개선.

가설: K_PS=15는 데이터 적은 fold(2019-2021, 3시즌)에서 투수-시즌/투수x손 LOO head가
노이즈를 그대로 학습하게 만들어 공유트리 split이 시드마다 흔들린다.
K_PS를 50으로 올려(더 강한 축소) 같은 2시드 비교를 한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea15_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
K_PS_NEW = 50.0
SEEDS = [42, 7]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
same_hand = X["same_hand"].to_numpy(np.float64)

CAT_PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val}  (K_PS={K_PS_NEW}) =====")
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
    log(f"  v35local={sc(v35l):.2f}")

    n_pitcher_seasons = meta.loc[tr_m].groupby(["pitcher_id", "season"]).ngroups
    log(f"  train 투수-시즌 조합 개수={n_pitcher_seasons}")

    p_seeds = []
    for seed in SEEDS:
        f_out = f"{CD}/{tag}_multires_kps{K_PS_NEW}_s{seed}.npy"
        if os.path.exists(f_out):
            p = np.load(f_out)
        else:
            sub_tr = meta.loc[tr_m, ["pitcher_id"]].copy()
            sub_tr["season"] = seasons[tr_m]
            sub_tr["sh"] = same_hand[tr_m]
            sub_tr["y"] = y[tr_m]
            ps = sub_tr.groupby(["pitcher_id", "season"])["y"].agg(s="sum", n="count")
            sub_tr = sub_tr.join(ps, on=["pitcher_id", "season"])
            g_tr = float(sub_tr["y"].mean())
            h1_tr = ((sub_tr["s"] - sub_tr["y"]) + K_PS_NEW * g_tr) / ((sub_tr["n"] - 1) + K_PS_NEW)
            psh = sub_tr.groupby(["pitcher_id", "season", "sh"])["y"].agg(s2="sum", n2="count")
            sub_tr = sub_tr.join(psh, on=["pitcher_id", "season", "sh"])
            h2_tr = ((sub_tr["s2"] - sub_tr["y"]) + K_PS_NEW * h1_tr) / ((sub_tr["n2"] - 1) + K_PS_NEW)
            h1_tr = h1_tr.to_numpy(np.float64); h2_tr = h2_tr.to_numpy(np.float64)
            log(f"    s{seed}: head1 std={h1_tr.std():.4f}  head2 std={h2_tr.std():.4f}")
            Ymat_tr = np.column_stack([y[tr_m], h1_tr, h2_tr])
            ts = time.time()
            n_es = int(tr_m.sum() * 0.92)
            m = CatBoostRegressor(**CAT_PARAMS, random_seed=seed)
            m.fit(X.loc[tr_m].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[tr_m][:n_es],
                 eval_set=(X.loc[tr_m].iloc[n_es:], Ymat_tr[n_es:]))
            p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
            np.save(f_out, p)
            log(f"    s{seed} 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)  단독={sc(p):.2f}")
        p_seeds.append(p)

    spread = max(sc(p) for p in p_seeds) - min(sc(p) for p in p_seeds)
    p_avg = np.mean(p_seeds, axis=0)
    for wv in [0.1, 0.15, 0.2]:
        blend = (1 - wv) * v35l + wv * p_avg
        results.setdefault(tag, {})[f"w{wv}"] = sc(blend) - sc(v35l)
    results[tag]["spread"] = spread
    results[tag]["v35local"] = sc(v35l)
    log(f"  fold {tag}: 시드폭(K_PS={K_PS_NEW})={spread:.2f}  (기존 K_PS=15 시드폭: A=0.68, C=93.04)")
    for wv in [0.1, 0.15, 0.2]:
        log(f"    w={wv} 이득={results[tag][f'w{wv}']:+.2f}")

print()
print("=" * 90)
print(f"{'fold':<6}{'K_PS=15시드폭(기존)':>20}{'K_PS=50시드폭':>16}{'w=0.15이득':>12}")
old_spread = {"A": 0.68, "C": 93.04}
for tag in ["A", "C"]:
    print(f"{tag:<6}{old_spread[tag]:20.2f}{results[tag]['spread']:16.2f}{results[tag]['w0.15']:12.2f}")
pd.DataFrame(results).T.to_csv("idea15_results.csv")
log(f"총 {time.time()-t0:.0f}s")
