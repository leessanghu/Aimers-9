"""idea40 — idea38(콜업)/idea39(피로·스트릭) 생존 후보 전원 stage2+stage3 검증.

stage1 결과 요약 (kσ = sqrt(g*n/1e5), 1단계 관대기준 3.2σ):
  streak_hot_flag      3-7월 4.07σ  Spearman 0.507  <- 최고
  streak_absdev5       3-7월 3.75σ  Spearman 0.921  (중복 큼)
  streak_dev5_sq       3-7월 3.18σ  Spearman 0.921  (중복 큼)
  fat_prev_pitches     3-7월 2.62σ  Spearman 0.573
  streak_cold_flag     3-7월 2.58σ  Spearman 0.505
  fat_roll3_pitches    3-7월 2.27σ  Spearman 0.586
  cu_density           3-7월 1.65σ  Spearman 0.404
  cu_lowdensity_x_prog 3-7월 2.12σ (부호 음)

stage2 = split-half 안정성 (검증행을 홀/짝으로 나눠 partial_gain 부호·크기 일치 확인)
stage3 = 162피처 + 후보블록으로 base HGB 재학습, 3시드, 월매칭(3-7월) 주판정
         baseline은 screen_v2_cache 재사용(동일 config).
게이트(스크리닝 v2): 3-7월Δ > +3 AND Δ > 시드폭 AND 전체Δ > -3
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea42_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7, 2024]
BASE_HGB = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20, max_depth=6, max_leaf_nodes=31)


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def partial_gain(y, p, z):
    z = np.nan_to_num(np.asarray(z, np.float64), nan=0.0)
    if z.std() == 0:
        return 0.0, 0.0
    A = np.column_stack([np.ones(len(y)), p])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rz = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


log("로드 + Marcel 후보 구성...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
raw = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                  usecols=["season", "game_month", "pitcher_id", "control_success"])
y = raw["control_success"].to_numpy(np.float64)
seasons = raw["season"].to_numpy(np.float64)
mo = raw["game_month"].to_numpy()
g_rate = float(y.mean())
sr = sorted(np.unique(seasons).tolist())

ys = raw.groupby(["pitcher_id", "season"])["control_success"].agg(s="sum", n="count")
sp = ys["s"].unstack().reindex(columns=sr)
npv = ys["n"].unstack().reindex(columns=sr)
S_, N_ = np.nan_to_num(sp.to_numpy()), np.nan_to_num(npv.to_numpy())
F = pd.DataFrame(index=X.index)
idxm = pd.MultiIndex.from_arrays([raw["pitcher_id"], raw["season"]])
career = X["asof_pitcher_success_rate_smooth"].to_numpy()
for tagname, wts, KM in [("m543", [(1,5.0),(2,4.0),(3,3.0)], 600.0),
                          ("m531", [(1,5.0),(2,3.0),(3,1.0)], 600.0),
                          ("m543k300", [(1,5.0),(2,4.0),(3,3.0)], 300.0)]:
    mS = np.zeros_like(S_, dtype=np.float64); mN = np.zeros_like(N_, dtype=np.float64)
    for j in range(len(sr)):
        for lag, wt in wts:
            if j - lag >= 0:
                mS[:, j] += wt * S_[:, j - lag]; mN[:, j] += wt * N_[:, j - lag]
    marcel = (mS + KM * g_rate) / (mN + KM)
    mt = pd.DataFrame(marcel, index=sp.index, columns=sr).stack(future_stack=True)
    v = pd.Series(mt.reindex(idxm).to_numpy()).fillna(g_rate).to_numpy(np.float64)
    F["marcel_" + tagname] = v
    F["marceldev_" + tagname] = v - career
F = F.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0)
log(f"  후보 {list(F.columns)}")

va = seasons == 2024
b = np.mean([np.load(f"phase90_cache/A_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
h = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) * np.load(f"phase90_cache/A_snc_{n}.npy")
             for n in ["d6", "d8"]], axis=0)
mm = np.mean([np.load(f"idea13_cache/A_multires_s{k}.npy") for k in [42, 7]], axis=0)
oo = np.mean([np.load(f"idea13_cache/A_ordinal_s{k}.npy") for k in [42, 7]], axis=0)
ddm = np.mean([np.load(f"idea31_cache/A_midaxis_s{k}.npy") for k in [42, 7]], axis=0)
p_base = np.clip(0.20 * b + 0.40 * h + 0.10 * mm + 0.20 * oo + 0.10 * ddm, 1e-6, 1 - 1e-6)
yv = y[va]
mv = mo[va]
seg = (mv >= 3) & (mv <= 7)

# ================= stage2: split-half =================
print()
print("=" * 88)
print("STAGE 2 — split-half 안정성 (3-7월 구간을 홀/짝으로 분할, 부호·크기 일치 확인)")
print("=" * 88)
i37 = np.where(seg)[0]
hA, hB = i37[0::2], i37[1::2]
nh = len(hA)
print(f"{'후보':<24}{'halfA(σ)':>12}{'halfB(σ)':>12}{'부호일치':>10}{'판정':>10}")
stage2_pass = []
for c in F.columns:
    z = F[c].to_numpy()[va]
    gA, pA = partial_gain(yv[hA], p_base[hA], z[hA])
    gB, pB = partial_gain(yv[hB], p_base[hB], z[hB])
    kA, kB = np.sqrt(gA * nh / 1e5), np.sqrt(gB * nh / 1e5)
    same = (pA > 0) == (pB > 0)
    ok = same and min(kA, kB) >= 1.5
    if ok:
        stage2_pass.append(c)
    print(f"{c:<24}{kA:12.2f}{kB:12.2f}{str(same):>10}{('통과' if ok else '기각'):>10}")
print(f"\nstage2 통과: {stage2_pass}")

# ================= stage3: 재학습 =================
BLOCKS = {
    "marcel_full": ["marcel_m543", "marceldev_m543"],
    "marcel_dev_only": ["marceldev_m543"],
    "marcel_multi": list(F.columns),
}
tr_m = seasons <= 2023
Xva = X.loc[va]


def sc(p, m_):
    yy = yv[m_]
    r = yy.mean(); BS = r * (1 - r)
    return 1e5 * (1 - np.mean((np.clip(p[m_], 0, 1) - yy) ** 2) / BS)


base_preds = [np.load(f"screen_v2_cache/A_baseline_s{s}.npy") for s in SEEDS]
base_avg = np.mean(base_preds, axis=0)
b37 = sc(base_avg, seg); ball = sc(base_avg, np.ones(len(mv), bool))
b37sp = max(sc(p, seg) for p in base_preds) - min(sc(p, seg) for p in base_preds)
print()
print("=" * 88)
print("STAGE 3 — 162피처 + 블록 재학습 (3시드, baseline은 screen_v2 캐시 재사용)")
print("=" * 88)
log(f"baseline  3-7월={b37:.2f}(폭{b37sp:.2f})  전체={ball:.2f}")

rows = []
for bn, cols in BLOCKS.items():
    if not cols:
        continue
    X2 = pd.concat([X, F[cols]], axis=1)
    preds = []
    for s in SEEDS:
        f = f"{CD}/A_{bn}_s{s}.npy"
        if os.path.exists(f):
            preds.append(np.load(f)); continue
        ts = time.time()
        m_ = HistGradientBoostingClassifier(**BASE_HGB, random_state=s)
        m_.fit(X2.loc[tr_m], y[tr_m], sample_weight=0.5 ** ((2023 - seasons[tr_m]) / 2.0))
        p = m_.predict_proba(X2.loc[va])[:, 1]
        np.save(f, p); preds.append(p)
        log(f"    [{bn}/s{s}] +{len(cols)}피처 iters={m_.n_iter_} ({time.time()-ts:.0f}s)")
    avg = np.mean(preds, axis=0)
    s37 = sc(avg, seg); sall = sc(avg, np.ones(len(mv), bool))
    sp = max(sc(p, seg) for p in preds) - min(sc(p, seg) for p in preds)
    rows.append((bn, len(cols), s37 - b37, sp, sall - ball))
    log(f"  {bn:<10}(+{len(cols)}) 3-7월Δ={s37-b37:+.2f}(폭{sp:.2f}) 전체Δ={sall-ball:+.2f}")

print()
print(f"{'블록':<12}{'피처수':>7}{'3-7월Δ':>10}{'시드폭':>8}{'전체Δ':>10}{'판정':>26}")
for bn, nc, d37, sp, dall in rows:
    ok = d37 > 3 and d37 > sp and dall > -3
    print(f"{bn:<12}{nc:7d}{d37:+10.2f}{sp:8.2f}{dall:+10.2f}{('통과' if ok else '기각'):>26}")
print("\n게이트: 3-7월Δ>+3 AND Δ>시드폭 AND 전체Δ>-3")
log(f"총 {time.time()-t0:.0f}s")
