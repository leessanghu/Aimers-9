"""idea49b — 축6(feature_weights) 재설계. idea49의 버그 수정.

버그1: "ability" 부분문자열이 form_reliability(reli-ABILITY)까지 오염 -> 제외.
버그2: weight=0.5 실측검증(100k행) 결과 importance 3.227->0.000 (거의 완전배제).
       "절반만 사용"이 아니라 "경쟁피처 있으면 사실상 배제"였음.
       0.5/2.0 -> **0.8/1.25**(완만한 우대)로 재설계.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea49_cache"
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
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def recover(col):
    c = np.round(meta[col].fillna(0).to_numpy(np.float64) * n_)
    co = c[order]
    d = np.empty(len(meta)); d[:-1] = co[1:] - co[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta)); lab[order] = d
    return lab


lab_rev = recover("asof_pitcher_reverse_rate")
lab_mid = recover("asof_pitcher_middle_rate")
valid = ~(np.isnan(lab_rev) | np.isnan(lab_mid))
tot = y + lab_rev + lab_mid
lab_other = np.where(valid, (tot == 0).astype(np.float64), np.nan)
h1 = np.where(valid, 1.0 - lab_mid, np.nan)
h2 = np.where(valid, 1.0 - lab_other, np.nan)

tr_m = seasons <= 2023
va_m = seasons == 2024
yv = y[va_m]
r = yv.mean(); BS = r * (1 - r)
w = 0.5 ** ((2023 - seasons) / 2.0)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)
A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)

base = A([f"phase90_cache/A_base_{n}.npy" for n in ["d6", "d8", "sub"]])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
mr = A([f"idea13_cache/A_multires_s{k}.npy" for k in [42, 7]])
od = A([f"idea13_cache/A_ordinal_s{k}.npy" for k in [42, 7]])
REST = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
uni_base = A([f"idea46_cache/A_midother_s{k}.npy" for k in [42, 7]])
v60a = lambda u, wu=0.20: sc((1 - wu) * REST + wu * u)
BASELINE = v60a(uni_base)
log(f"v60a 로컬 기준선 = {BASELINE:.2f}")

Ymat_tr = np.column_stack([y[tr_m], h1[tr_m], h2[tr_m]])
n_es = int(tr_m.sum() * 0.92)

# 버그1 수정: 정확히 "ability" 단어 경계로만 매칭 (reliability 등 부분문자열 오염 제거)
shrink_cols = [c for c in X.columns if any(
    c == k or c.startswith(k + "_") or c.endswith("_" + k) or f"_{k}_" in c
    for k in ["ability", "x_ability", "asof_pitcher_success_rate", "inseason_success"])]
boost_cols = [c for c in X.columns if "middle" in c]
assert "form_reliability" not in shrink_cols, "버그1 재발"
log(f"  축소대상({len(shrink_cols)}): {shrink_cols}")
log(f"  증폭대상({len(boost_cols)}개)")

for tag, sv, bv in [("mild", 0.8, 1.25), ("verymild", 0.9, 1.10)]:
    fw = {c: sv for c in shrink_cols}
    fw.update({c: bv for c in boost_cols})
    ps = []
    for seed in SEEDS:
        f = f"{CD}/A_featweight_{tag}_s{seed}.npy"
        if os.path.exists(f):
            ps.append(np.load(f)); continue
        ts = time.time()
        params = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                     loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50,
                     random_seed=seed, feature_weights=fw)
        m = CatBoostRegressor(**params)
        m.fit(X.loc[tr_m].iloc[:n_es], Ymat_tr[:n_es], sample_weight=w[tr_m][:n_es],
              eval_set=(X.loc[tr_m].iloc[n_es:], Ymat_tr[n_es:]))
        p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
        np.save(f, p); ps.append(p)
        log(f"    [{tag}/s{seed}] best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    u = np.mean(ps, axis=0)
    v = v60a(u)
    print(f"  {tag}(축소x{sv}/증폭x{bv})  단독={sc(u):.2f}  v60a로컬={v:.2f}  Δ={v-BASELINE:+.2f}")

log(f"총 {time.time()-t0:.0f}s")
