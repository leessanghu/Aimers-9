"""idea52 — v60b 검증용 fold A 로컬 테스트. midother(2-head)를 3-7월 행만으로
학습(train<=2023, 3-7월만) -> 2024 전체 예측, v60a와 비교.
featcache 재사용(빠름). 실측 판단 전 사전 체크용.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea52_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
mo = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                 usecols=["game_month"])["game_month"].to_numpy()
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
tr37_m = tr_m & (mo >= 3) & (mo <= 7)
yv = y[va_m]; mv = mo[va_m]; seg37 = (mv >= 3) & (mv <= 7)
r = yv.mean(); BS = r * (1 - r)
w_full = 0.5 ** ((2023 - seasons) / 2.0)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)
A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)

log(f"  train전체 {tr_m.sum():,}행 -> 3-7월만 {tr37_m.sum():,}행 ({tr37_m.sum()/tr_m.sum()*100:.1f}%)")

base = A([f"phase90_cache/A_base_{n}.npy" for n in ["d6", "d8", "sub"]])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
mr = A([f"idea13_cache/A_multires_s{k}.npy" for k in [42, 7]])
od = A([f"idea13_cache/A_ordinal_s{k}.npy" for k in [42, 7]])
REST = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
uni_full = A([f"idea46_cache/A_midother_s{k}.npy" for k in [42, 7]])  # 기존(전체시즌 학습) midother
v60a = lambda u, wu=0.20: sc((1 - wu) * REST + wu * u)
BASE_FULL = v60a(uni_full)
log(f"v60a(midother=전체시즌학습) 로컬 기준선 = {BASE_FULL:.2f} (실측 1080.60)")

CAT = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)
Ymat = np.column_stack([y, h1, h2])
n_es = int(tr37_m.sum() * 0.92)

ps = []
for seed in SEEDS:
    f = f"{CD}/A_midother37_s{seed}.npy"
    if os.path.exists(f):
        ps.append(np.load(f)); continue
    ts = time.time()
    m = CatBoostRegressor(**CAT, random_seed=seed)
    m.fit(X.loc[tr37_m].iloc[:n_es], Ymat[tr37_m][:n_es], sample_weight=w_full[tr37_m][:n_es],
          eval_set=(X.loc[tr37_m].iloc[n_es:], Ymat[tr37_m][n_es:]))
    p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
    np.save(f, p); ps.append(p)
    log(f"    s{seed} best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")

uni37 = np.mean(ps, axis=0)
spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
print()
print("=" * 78)
print("midother 학습범위 비교: 전체시즌 vs 3-7월만 (동일 fold A, 동일 나머지구성)")
print("=" * 78)
print(f"  전체시즌학습(기존)  단독={sc(uni_full):.2f}  v60a로컬={BASE_FULL:.2f}")
print(f"  3-7월만학습(신규)   단독={sc(uni37):.2f}  시드폭={spread:.2f}  v60a로컬={v60a(uni37):.2f}  "
      f"Δ={v60a(uni37)-BASE_FULL:+.2f}")
print()
def sc37(u):
    p = 0.8 * REST + 0.2 * u
    yy = yv[seg37]; rr = yy.mean(); bb = rr * (1 - rr)
    return 1e5 * (1 - np.mean((np.clip(p[seg37], 0, 1) - yy) ** 2) / bb)
print(f"  3-7월구간에서: 전체시즌학습={sc37(uni_full):.2f}  3-7월만학습={sc37(uni37):.2f}  "
     f"Δ={sc37(uni37)-sc37(uni_full):+.2f}")
log(f"총 {time.time()-t0:.0f}s")
