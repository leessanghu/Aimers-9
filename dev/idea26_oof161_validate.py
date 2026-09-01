"""3단계 — OOF161(batter_id+season+pitcher_hand) 3폴드+시드반복 정식검증.
2단계 split-half 통과(+24.45/+23.35, 판정기준 +9.10) 유일 생존자.
기존 162피처 + OOF161 1개 추가해서 HGB classifier 재학습, fold A/C/B 2시드.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea26_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
K_OOF161 = 20.0


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

log("train.csv에서 batter_id, pitcher_hand 로드...")
raw = pd.read_csv("../data/train.csv", encoding="utf-8-sig", usecols=["row_id", "batter_id", "pitcher_hand"])
raw["row_num"] = raw["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
raw = raw.set_index("row_num").loc[meta["row_num"].to_numpy()].reset_index(drop=True)
assert len(raw) == len(meta)
batter_id = raw["batter_id"].to_numpy()
pitcher_hand = raw["pitcher_hand"].to_numpy()

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

    # OOF161 구축: train 데이터에서만, (batter_id,season,pitcher_hand) 누적(자기제외), 전체행에 적용
    sub = pd.DataFrame({"bid": batter_id, "season": seasons, "ph": pitcher_hand, "y": y})
    g_tr = float(y[tr_m].mean())
    grp = sub.groupby(["bid", "season", "ph"])["y"]
    cs = grp.cumsum() - sub["y"]
    cn = grp.cumcount()
    oof161 = ((cs + K_OOF161 * g_tr) / (cn + K_OOF161)).to_numpy(np.float64)

    X163 = X.copy()
    X163["oof161_batter_season_hand"] = oof161

    p_seeds = []
    for seed in SEEDS:
        f_out = f"{CD}/{tag}_d6plus161_s{seed}.npy"
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
    log(f"  d6+oof161(163피처) 2시드평균 단독={sc(p_avg):.2f}  (기존d6대비 {sc(p_avg)-sc(d6_orig):+.2f})  시드폭={spread:.2f}")

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
