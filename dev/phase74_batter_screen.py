"""타자 조건부 후보 — 축소계수 K 실측 + 증분 잠재력 스크리닝.

count_split 때와 동일한 절차:
  1) 타자 자신의 marginal 대비 조건부 편차의 '노이즈보정 진짜 SD'를 실측
  2) K = p(1-p)/Var(진짜편차) 로 축소계수 산출
  3) 그 K로 피처를 만들어 GBDT(132피처) 위 증분 잠재력 측정
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

import batter_split as bs

VALID_SEASON = 2024
CACHE = "phase67_cache"
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def _clean(z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    return z


def partial_gain(y, p, z):
    z = _clean(z)
    if z.std() == 0:
        return 0.0, 0.0
    A = np.column_stack([np.ones(len(y)), p])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rz = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


def block_gain(y, p, Z):
    Z = np.column_stack([_clean(Z[:, j]) for j in range(Z.shape[1])])
    keep = [j for j in range(Z.shape[1]) if Z[:, j].std() > 0]
    Z = Z[:, keep]
    n, k = len(y), Z.shape[1]
    X0 = np.column_stack([np.ones(n), p])
    X1 = np.column_stack([X0, Z])

    def r2(X):
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - (y - X @ c).var() / y.var()
    return 1e5 * ((r2(X1) - r2(X0)) - k / n), k


def estimate_K(df, keys, label="", global_p=None):
    """조건부 편차의 노이즈보정 진짜 SD -> K = p(1-p)/Var."""
    bm = df.groupby("batter_id")["control_success"].mean()
    d = df.assign(_prior=df["batter_id"].map(bm))
    g = d.groupby(keys)["control_success"]
    cm, cn = g.mean(), g.size()
    pri = d.groupby(keys)["_prior"].first()
    dev = cm - pri
    w = cn / cn.sum()
    var_obs = float((w * dev ** 2).sum())
    noise = float((w * cm * (1 - cm) / cn).sum())
    var_true = max(1e-12, var_obs - noise)
    p = global_p if global_p is not None else float(df["control_success"].mean())
    K = p * (1 - p) / var_true
    log(f"  {label}: cells={len(cm):,} median_n={cn.median():.0f} "
        f"raw_SD={dev.std():.4f} true_SD={var_true**0.5:.5f} -> K={K:.0f}")
    return K


log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
df["count_state"] = df["balls_before"] * 4 + df["strikes_before"]
g_rate = float(df["control_success"].mean())
g_mid = float(df["asof_pitcher_middle_rate"].mean(skipna=True))
g_bmid = float(df["asof_batter_middle_rate"].mean(skipna=True))
sr = sorted(df["season"].unique().tolist())
d = df[df["season"] == VALID_SEASON].copy().reset_index(drop=True)

p_ctrl = np.load(f"{CACHE}/gbdt_v26_valid_pred.npy")
y = d["control_success"].to_numpy(np.float64)
log(f"n={len(y):,}  1시그마={1e5/len(y):.2f}점")

log("\n축소계수 K 실측 (노이즈보정)...")
K_bc = estimate_K(df, ["batter_id", "count_state"], "batter x count", g_rate)
K_bp = estimate_K(df, ["batter_id", "pitcher_hand"], "batter x pitcher_hand", g_rate)

log("\n피처 생성...")
marg = bs.build_batter_marginal(df)
prior = bs.lookup_batter_prior(d, marg, sr, g_rate)

X_bc = bs.transform_bcount(d, bs.build_bcount_table(df), prior, sr, k=K_bc)
X_bp = bs.transform_bplatoon(d, bs.build_bplatoon_table(df), prior, sr, k=K_bp)
X_bm = bs.transform_batter_middle(d, bs.build_batter_middle_table(df), sr, g_bmid)

blocks = {
    "batter x count": X_bc,
    "batter x pitcher_hand": X_bp,
    "batter in-season middle": X_bm,
}

log("\n" + "=" * 78)
log("증분 잠재력 (GBDT 132피처 위, 1시그마=0.39점)")
log("=" * 78)
print(f"{'피처':<30}{'증분점수':>10}{'부분상관':>11}{'시그마':>9}")
print("-" * 62)
allX = []
for name, X in blocks.items():
    for c in X.columns:
        gn, pc = partial_gain(y, p_ctrl, X[c].to_numpy())
        print(f"{c:<30}{gn:10.2f}{pc:+11.4f}{abs(pc)*np.sqrt(len(y)):9.1f}")
    allX.append(X)
print()
for name, X in blocks.items():
    gb, k = block_gain(y, p_ctrl, X.to_numpy(np.float64))
    print(f"  [{name}] 블록 증분 = {gb:6.1f}  (피처 {X.shape[1]})")

X_all = pd.concat(allX, axis=1)
gb_all, k_all = block_gain(y, p_ctrl, X_all.to_numpy(np.float64))
print(f"\n  [타자 전체 합동] 증분 = {gb_all:.1f}  (피처 {X_all.shape[1]})")
print(f"  실현율 0.6 적용 예상 실측 = {gb_all*0.6:+.1f}점")

print()
print("  비교 (지금까지 최강):")
print("    bat_inseason_smooth (v27)  17.1  6.6시그마")
print("    count_diff (v26)            9.3  4.9시그마")
print("    inseason_middle (phase73)   2.8  2.7시그마")

log(f"\n실측 K 값 -> batter_split.py 반영 필요: K_BCOUNT={K_bc:.0f}  K_BPLATOON={K_bp:.0f}")
log(f"총 {time.time()-t0:.0f}s")
