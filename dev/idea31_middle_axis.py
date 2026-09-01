"""아이디어 — middle 라벨을 독립 auxiliary head로 셰어드트리에 추가.
SHAP vs partial_gain 대조에서 middle 계열 피처(inseason_middle_smooth,
prev3/5_game_middle_rate, form3/5_middle)들이 일관되게 "신호는 있는데 SHAP비중은
낮음" 패턴을 보임 -- middle이 지금까지 Hurdle(core_fail=reverse OR middle 묶음)/
ordinal(순차분해)에서만 쓰였지 독립 신호로는 한 번도 안 씀.
head0=y, head1=(1-lab_middle) 셰어드트리, 전체표본(복원가능행) 사용.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea31_cache"
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
pid = meta["pitcher_id"].to_numpy()

log("투구단위 middle 라벨 복원...")
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])

def _diff_label(rate_col):
    c = np.round(meta[rate_col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(len(meta))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta))
    lab[order] = d
    return lab

lab_middle = _diff_label("asof_pitcher_middle_rate")
valid_lab = ~np.isnan(lab_middle)
log(f"  라벨 유효행 {valid_lab.sum()}/{len(meta)} ({valid_lab.mean()*100:.2f}%)")

CAT_PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
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

    Ymat_tr = np.column_stack([y[tr_m], 1.0 - lab_middle[tr_m]])

    p_seeds = []
    for seed in SEEDS:
        f_out = f"{CD}/{tag}_midaxis_s{seed}.npy"
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
            log(f"    s{seed} 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)  단독={sc(p):.2f}")
        p_seeds.append(p)

    p_avg = np.mean(p_seeds, axis=0)
    spread = max(sc(p) for p in p_seeds) - min(sc(p) for p in p_seeds)
    for wv in [0.1, 0.15, 0.2]:
        blend = (1 - wv) * v35l + wv * p_avg
        results.setdefault(tag, {})[f"w{wv}"] = sc(blend) - sc(v35l)
    results[tag]["solo"] = sc(p_avg)
    results[tag]["spread"] = spread
    log(f"  midaxis 2시드평균 단독={sc(p_avg):.2f}  시드폭={spread:.2f}")
    for wv in [0.1, 0.15, 0.2]:
        log(f"    w={wv} 이득={results[tag][f'w{wv}']:+.2f}")

print()
print("=" * 90)
print(f"{'fold':<6}{'단독':>10}{'시드폭':>8}" + "".join(f"{'w='+str(w):>9}" for w in [0.1, 0.15, 0.2]))
for tag, r in results.items():
    print(f"{tag:<6}{r['solo']:10.2f}{r['spread']:8.2f}" + "".join(f"{r[f'w{w}']:+9.2f}" for w in [0.1, 0.15, 0.2]))
gain_a = max(results["A"][f"w{w}"] for w in [0.1, 0.15, 0.2])
print(f"\n[신기준] 주검증 fold A 최고이득={gain_a:+.2f}  {'양수->통과후보' if gain_a>0 else '음수->기각'}")
log(f"총 {time.time()-t0:.0f}s")
