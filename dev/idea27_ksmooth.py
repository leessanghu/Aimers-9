"""in-season 스무딩 K=15 vs 60 재검증 (phase31은 v12 구성/단일폴드였음).
현재 162피처 구성에서도 유효한지 fold A/C 2시드로 확인.
K는 inseason의 3개 피처(success/ball/reverse smooth)에만 영향.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import inseason as INS_MOD
from inseason import build_season_end_table, transform_inseason

CD = "idea27_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
KS = [15.0, 60.0]
COLS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth"]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 + 원본 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
df = df.set_index("row_num").loc[meta["row_num"].to_numpy()].reset_index()
assert (df["control_success"].to_numpy() == y).all(), "행 정렬 불일치"
g = float(y.mean())
sr = sorted(df["season"].unique().tolist())
se = build_season_end_table(df)

log("K별 inseason 피처 생성...")
feats = {}
for K in KS:
    INS_MOD.K_SMOOTH = K
    ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
    feats[K] = ins[COLS].to_numpy(np.float64)
    log(f"  K={K}: {COLS[0]} std={feats[K][:,0].std():.4f} (K클수록 수축->std 작아짐)")
assert not np.allclose(feats[15.0], feats[60.0]), "K가 실제로 반영 안 됨"

HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
           early_stopping=True, validation_fraction=0.1, n_iter_no_change=20)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)
    sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    d8 = np.load(f"phase90_cache/{tag}_base_d8.npy")
    subm = np.load(f"phase90_cache/{tag}_base_sub.npy")
    base_orig = np.mean([np.load(f"phase90_cache/{tag}_base_d6.npy"), d8, subm], axis=0)
    v35l = 0.55 * base_orig + 0.45 * hur
    log(f"  v35local={sc(v35l):.2f}")

    row = {"v35local": sc(v35l)}
    for K in KS:
        Xk = X.copy()
        for j, c in enumerate(COLS):
            Xk[c] = feats[K][:, j]
        ps = []
        for seed in SEEDS:
            f = f"{CD}/{tag}_K{int(K)}_s{seed}.npy"
            if os.path.exists(f):
                p = np.load(f)
            else:
                ts = time.time()
                m = HistGradientBoostingClassifier(**HGB, random_state=seed).fit(
                    Xk.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
                p = m.predict_proba(Xk.loc[va_m])[:, 1]
                np.save(f, p)
                log(f"    K={K} s{seed} 완료 iters={m.n_iter_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
            ps.append(p)
        avg = np.mean(ps, axis=0)
        spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
        blend = 0.55 * np.mean([avg, d8, subm], axis=0) + 0.45 * hur
        row[f"K{int(K)}_solo"] = sc(avg)
        row[f"K{int(K)}_blend"] = sc(blend)
        row[f"K{int(K)}_spread"] = spread
        log(f"  K={K}: d6단독(2시드)={sc(avg):.2f} 시드폭={spread:.2f}  블렌드={sc(blend):.2f}")
    results[tag] = row
    log(f"  >>> K60 - K15 : 단독 {row['K60_solo']-row['K15_solo']:+.2f}   블렌드 {row['K60_blend']-row['K15_blend']:+.2f}")

print()
print("=" * 90)
print(f"{'fold':<6}{'K15단독':>10}{'K60단독':>10}{'차이':>8}{'K15블렌드':>11}{'K60블렌드':>11}{'차이':>8}{'시드폭max':>10}")
for tag, r in results.items():
    sp = max(r["K15_spread"], r["K60_spread"])
    print(f"{tag:<6}{r['K15_solo']:10.2f}{r['K60_solo']:10.2f}{r['K60_solo']-r['K15_solo']:+8.2f}"
          f"{r['K15_blend']:11.2f}{r['K60_blend']:11.2f}{r['K60_blend']-r['K15_blend']:+8.2f}{sp:10.2f}")
gains = [results[t]["K60_blend"] - results[t]["K15_blend"] for t in ["A", "C"]]
sp = max(max(results[t]["K15_spread"], results[t]["K60_spread"]) for t in ["A", "C"])
print(f"\n클린폴드 최소이득={min(gains):+.2f}  시드폭최대={sp:.2f}  "
      f"{'신뢰가능' if min(gains) > sp else '신뢰불가(노이즈 이하)'}")
print("주의: 이 검증은 base3 중 d6 하나만 교체한 것 -> 전체 적용시 효과는 달라질 수 있음")
log(f"총 {time.time()-t0:.0f}s")
