"""아이디어G — ordinal 캐스케이드에서 Hurdle과 겹치는 stage1x2(core_fail 재추정)는 버리고,
stage3(P(success|no core_fail), Hurdle의 succ_nc_model에 대한 독립적 두번째 추정)만 블렌드.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea19_cache"
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

log("투구단위 라벨 복원...")
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

lab_reverse = _diff_label("asof_pitcher_reverse_rate")
lab_middle = _diff_label("asof_pitcher_middle_rate")
valid_lab = ~(np.isnan(lab_reverse) | np.isnan(lab_middle))

HGB_CLS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
              early_stopping=True, validation_fraction=0.08, n_iter_no_change=20)

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
    log(f"  v35local={sc(v35l):.2f}")

    p3_seeds = []
    for seed in SEEDS:
        f_out = f"{CD}/{tag}_stage3_only_s{seed}.npy"
        if os.path.exists(f_out):
            p3 = np.load(f_out)
        else:
            not_rev_mid = tr_m & valid_lab & (lab_reverse == 0) & (lab_middle == 0)
            ts = time.time()
            m3 = HistGradientBoostingClassifier(**HGB_CLS, random_state=seed)
            m3.fit(X.loc[not_rev_mid], y[not_rev_mid], sample_weight=w[not_rev_mid])
            p3 = m3.predict_proba(X.loc[va_m])[:, 1]
            np.save(f_out, p3)
            log(f"    s{seed} 학습완료 ({time.time()-ts:.0f}s)  단독={sc(p3):.2f}")
        p3_seeds.append(p3)

    p3_avg = np.mean(p3_seeds, axis=0)
    spread = max(sc(p) for p in p3_seeds) - min(sc(p) for p in p3_seeds)
    log(f"  stage3단독 시드폭={spread:.2f}")

    # Hurdle의 succ_nc_model 부분을 stage3(독립 재추정)와 평균 -- 겹치는 core_fail쪽은 그대로 Hurdle 것만 사용
    snc_avg = np.mean([np.load(f"phase90_cache/{tag}_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
    core_avg = np.mean([np.load(f"phase90_cache/{tag}_core_{n}.npy") for n in ["d6", "d8"]], axis=0)
    for wv in [0.0, 0.2, 0.3, 0.4, 0.5]:
        snc_blend = (1 - wv) * snc_avg + wv * p3_avg
        hur_v2 = (1 - core_avg) * snc_blend
        blend_full = 0.40 * base3 + 0.45 * hur_v2 + 0.15 * np.load(f"idea13_cache/{tag}_multires_s42.npy")
        results.setdefault(tag, {})[f"w{wv}"] = sc(blend_full)
    results[tag]["v35local"] = sc(v35l)
    results[tag]["spread"] = spread
    for wv in [0.0, 0.2, 0.3, 0.4, 0.5]:
        log(f"    succ_nc를 stage3와 {wv}:{1-wv} 혼합 -> v40스타일블렌드={results[tag][f'w{wv}']:.2f}  "
           f"(w=0기준대비 {results[tag][f'w{wv}']-results[tag]['w0.0']:+.2f})")

print()
print("=" * 90)
print(f"{'fold':<6}{'stage3시드폭':>12}" + "".join(f"{'w='+str(w):>10}" for w in [0.0, 0.2, 0.3, 0.4, 0.5]))
for tag, r in results.items():
    print(f"{tag:<6}{r['spread']:12.2f}" + "".join(f"{r[f'w{w}']:10.2f}" for w in [0.0, 0.2, 0.3, 0.4, 0.5]))
for wv in [0.2, 0.3, 0.4, 0.5]:
    gains_clean = [results[t][f"w{wv}"] - results[t]["w0.0"] for t in ["A", "C"]]
    gain_b = results["B"][f"w{wv}"] - results["B"]["w0.0"]
    print(f"  succ_nc를 stage3와 {wv} 혼합: 클린폴드 최소이득={min(gains_clean):+.2f}  (참고 B={gain_b:+.2f})")
pd.DataFrame(results).T.to_csv("idea19_results.csv")
log(f"총 {time.time()-t0:.0f}s")
