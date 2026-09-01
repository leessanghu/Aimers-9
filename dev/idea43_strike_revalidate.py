"""strike축 재검증 — idea32(2시드, fold A만)의 결함 보완.
원본: fold A w0.1=-0.04, 시드폭=17.90 (점추정치보다 시드폭이 10배 이상 커서
사실상 무의미한 측정이었음). SHAP순위 87위/splits 5회로 middle(122위/2회)에
근접한 미활용 축 -> 미활용도 규칙상 기대값 높음. 3번째 시드 + fold C 추가.
구성은 idea32와 완전 동일(head1=1-lab_strike, MultiRMSEWithMissingValues).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea32_cache"
t0 = time.time()
SEEDS = [42, 7, 2024]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def recover(rate_col):
    c = np.round(meta[rate_col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(len(meta))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta))
    lab[order] = d
    return lab


lab_strike = recover("asof_pitcher_strike_rate")
valid_lab = ~np.isnan(lab_strike)
log(f"  유효행 {valid_lab.sum():,} 발생률={np.nanmean(lab_strike)*100:.2f}%")

CAT_PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = (seasons <= upto) & valid_lab
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

    Ymat_tr = np.column_stack([y[tr_m], 1.0 - lab_strike[tr_m]])
    p_seeds = []
    for seed in SEEDS:
        f_out = f"{CD}/{tag}_strike_s{seed}.npy"
        if os.path.exists(f_out):
            p = np.load(f_out)
        else:
            ts = time.time()
            n_es = int(tr_m.sum() * 0.92)
            m = CatBoostRegressor(**CAT_PARAMS, random_seed=seed)
            m.fit(X.loc[tr_m].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[tr_m][:n_es],
                 eval_set=(X.loc[tr_m].iloc[n_es:], Ymat_tr[n_es:]))
            p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
            np.save(f_out, p)
            log(f"    s{seed} 완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
        p_seeds.append(p)

    p_avg = np.mean(p_seeds, axis=0)
    spread = max(sc(p) for p in p_seeds) - min(sc(p) for p in p_seeds)
    for wv in [0.1, 0.15, 0.2]:
        blend = (1 - wv) * v35l + wv * p_avg
        results.setdefault(tag, {})[f"w{wv}"] = sc(blend) - sc(v35l)
    results[tag]["solo"] = sc(p_avg)
    results[tag]["spread"] = spread
    log(f"  fold {tag} (3시드): 단독={sc(p_avg):.2f} 시드폭={spread:.2f}  "
       f"w0.1={results[tag]['w0.1']:+.2f} w0.15={results[tag]['w0.15']:+.2f} w0.2={results[tag]['w0.2']:+.2f}")

print()
print("=" * 70)
print("strike축 재검증 (3시드, fold A+C)")
for tag, r in results.items():
    print(f"fold {tag}: 단독={r['solo']:.2f} 시드폭={r['spread']:.2f}  "
         f"w0.1={r['w0.1']:+.2f} w0.15={r['w0.15']:+.2f} w0.2={r['w0.2']:+.2f}")
print()
print("참고: 원본(2시드) fold A w0.1=-0.04(시드폭17.90) -> 이번 3시드 결과와 비교")
print("aux head 편향규칙: +5.9~6.64 보정. middle(122위/2회)과 유사 미활용도라 신뢰도 높음")
log(f"총 {time.time()-t0:.0f}s")
