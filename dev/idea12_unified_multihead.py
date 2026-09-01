"""아이디어D — 통합 다중head 공유트리 (v40 다중해상도 + v41 순서형분해 메커니즘 결합).

실측 검증된 두 성공의 공통점: 전체표본 유지 + 타겟쪽 재구성(엔티티 셀 안 쪼갬).
차이점: v40(multires)은 '같은 양의 해상도 다양화'로 트리구조만 정규화, v41(ordinal)은
'서로 다른 사건 순차분해'로 정보를 얻음. 이 둘을 하나의 공유트리에 합친다:

  head0 = y (원본, 최종출력에 유일하게 쓰임)
  head1 = not_reverse (1-lab_reverse, 전체 유효행)
  head2 = not_middle | not_reverse (lab_reverse==1인 행은 NaN 마스킹, v39와 동일)
  head3 = 투수-시즌 LOO 성공률 (전체표본, 셀 큼)
  head4 = 투수x손(same_hand) LOO 성공률 (전체표본, 셀 큼)

*** 반드시 fold A/C 우선판정, fold B는 참고만, 시드 2개 반복. ***
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea12_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
K_PS = 15.0
SEEDS = [42, 7]
WEIGHTS = [0.1, 0.15, 0.2, 0.3]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
same_hand = X["same_hand"].to_numpy(np.float64) if "same_hand" in X.columns else \
    (meta.get("pitcher_hand", 0) == meta.get("batter_hand", 0)).astype(np.float64)

log("투구단위 라벨 복원 (reverse/middle, phase 캐시와 동일 로직)...")
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pos = pd.Series(np.arange(len(meta)), index=meta.index).loc[order].to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])

def _diff_label(rate_col):
    c = np.round(meta[rate_col].fillna(0).to_numpy(np.float64) * n_)
    lab = np.full(len(meta), np.nan)
    c_ord = c[order]
    d = np.empty(len(meta))
    d[:-1] = c_ord[1:] - c_ord[:-1]
    d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab_ord = d
    lab = np.empty(len(meta))
    lab[order] = lab_ord
    return lab

lab_reverse = _diff_label("asof_pitcher_reverse_rate")
lab_middle = _diff_label("asof_pitcher_middle_rate")
valid_lab = ~(np.isnan(lab_reverse) | np.isnan(lab_middle))
log(f"  라벨 유효행 {valid_lab.sum()}/{len(meta)} ({valid_lab.mean()*100:.2f}%)")

CAT_PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  random_seed=42, loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)

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
    row = dict(v35local=sc(v35l))
    log(f"  v35local={row['v35local']:.2f}")

    sub_tr = meta.loc[tr_m, ["pitcher_id"]].copy()
    sub_tr["season"] = seasons[tr_m]
    sub_tr["sh"] = same_hand[tr_m]
    sub_tr["y"] = y[tr_m]
    ps = sub_tr.groupby(["pitcher_id", "season"])["y"].agg(s="sum", n="count")
    sub_tr = sub_tr.join(ps, on=["pitcher_id", "season"])
    g_tr = float(sub_tr["y"].mean())
    h3_tr = ((sub_tr["s"] - sub_tr["y"]) + K_PS * g_tr) / ((sub_tr["n"] - 1) + K_PS)
    psh = sub_tr.groupby(["pitcher_id", "season", "sh"])["y"].agg(s2="sum", n2="count")
    sub_tr = sub_tr.join(psh, on=["pitcher_id", "season", "sh"])
    h4_tr = ((sub_tr["s2"] - sub_tr["y"]) + K_PS * h3_tr) / ((sub_tr["n2"] - 1) + K_PS)
    h3_tr = h3_tr.to_numpy(np.float64); h4_tr = h4_tr.to_numpy(np.float64)

    h1_tr = 1 - lab_reverse[tr_m]
    h2_tr = np.where(lab_reverse[tr_m] == 0, 1 - lab_middle[tr_m], np.nan)
    Ymat_tr = np.column_stack([y[tr_m], h1_tr, h2_tr, h3_tr, h4_tr])

    preds_seed = []
    for seed in SEEDS:
        fcache = f"{CD}/{tag}_head0_s{seed}.npy"
        if os.path.exists(fcache):
            p0 = np.load(fcache)
        else:
            ts = time.time()
            params = dict(CAT_PARAMS, random_seed=seed)
            m = CatBoostRegressor(**params)
            n_es = int(len(sub_tr) * 0.92)
            m.fit(X.loc[tr_m].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[tr_m][:n_es],
                 eval_set=(X.loc[tr_m].iloc[n_es:], Ymat_tr[n_es:]))
            heads_va = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)
            p0 = heads_va[:, 0]
            np.save(fcache, p0)
            log(f"    s{seed} 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)")
        preds_seed.append(p0)
        log(f"  s{seed}: head0단독={sc(p0):.2f}")
        for wv in WEIGHTS:
            blend = (1 - wv) * v35l + wv * p0
            row[f"s{seed}_w{wv}"] = sc(blend)

    p0_avg = np.mean(preds_seed, axis=0)
    seed_spread = max(sc(p) for p in preds_seed) - min(sc(p) for p in preds_seed)
    row["seed_spread"] = seed_spread
    log(f"  시드폭={seed_spread:.2f}")
    for wv in WEIGHTS:
        blend = (1 - wv) * v35l + wv * p0_avg
        row[f"avg_w{wv}"] = sc(blend)
        log(f"    v35local+avg2seed(w={wv}) = {row[f'avg_w{wv}']:.2f}  (v35l대비 {row[f'avg_w{wv}']-row['v35local']:+.2f})")
    results[tag] = row

print()
print("=" * 90)
print(f"{'fold':<6}{'v35local':>10}{'시드폭':>9}" + "".join(f"{'w='+str(w):>9}" for w in WEIGHTS))
for tag, r in results.items():
    print(f"{tag:<6}{r['v35local']:10.2f}{r['seed_spread']:9.2f}" +
         "".join(f"{r[f'avg_w{w}']:9.2f}" for w in WEIGHTS))
for wv in WEIGHTS:
    gains_clean = [results[t][f"avg_w{wv}"] - results[t]["v35local"] for t in ["A", "C"]]
    gain_b = results["B"][f"avg_w{wv}"] - results["B"]["v35local"]
    max_spread = max(results[t]["seed_spread"] for t in ["A", "C"])
    trustworthy = min(gains_clean) > max_spread
    print(f"  w={wv}: 클린폴드(A,C) 최소이득={min(gains_clean):+.2f}  시드폭최대={max_spread:.2f}  "
         f"{'신뢰가능' if trustworthy else '신뢰불가(시드폭 이하)'}  (참고 B={gain_b:+.2f})")
pd.DataFrame(results).T.to_csv("idea12_results.csv")
log(f"총 {time.time()-t0:.0f}s")
