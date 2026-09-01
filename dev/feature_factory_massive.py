"""Feature Factory 대규모 확대(B) — 카테고리형 컬럼 조합을 체계적으로 생성해
OOF expanding target-encoding 후보를 대량 스캔. NVIDIA 그랜드마스터 사례(10K생성)를
우리 인프라로 재현 가능한 규모(~2000-3000)로 축소.

leakage-safe: 각 행은 row_num 기준 '이전 행들'만 사용하는 전역 expanding 통계.
다중검정 보정: 후보 개수에 맞춰 Bonferroni 임계값을 계산해서 같이 출력.
"""
import itertools
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy import stats

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def partial_gain(y, p, z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    if z.std() == 0:
        return 0.0, 0.0
    A = np.column_stack([np.ones(len(y)), p])
    cy = np.linalg.lstsq(A, y, rcond=None)[0]
    cz = np.linalg.lstsq(A, z, rcond=None)[0]
    ry, rz = y - A @ cy, z - A @ cz
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
df = df.sort_values("row_num").reset_index(drop=True)
y_all = df["control_success"].to_numpy(np.float64)
g = float(y_all.mean())
va_m = (df["season"] == 2024).to_numpy()

p_base = np.load("phase90_cache/A_base_d6.npy")
assert p_base.shape[0] == va_m.sum()

# 이미 스크리닝된 것 제외(OOF001,011,021,031,041,051,061,071,081,091,101,111,121,131,141,151,161)
POOL = ["season", "game_type", "game_month", "game_dayofweek", "top_bottom", "inning",
       "balls_before", "strikes_before", "outs_before", "base_state", "pitcher_hand",
       "batter_hand", "num_runners_on"]
df["count_state"] = df["balls_before"] * 4 + df["strikes_before"]
POOL2 = POOL + ["count_state"]

log(f"조합 생성 중 (풀 크기={len(POOL2)})...")
combos = []
for r in [2, 3]:
    for c in itertools.combinations(POOL2, r):
        combos.append(list(c))
log(f"  2/3-way 조합 {len(combos)}개")

K_VALUES = [30.0]  # 시간절약을 위해 K 하나만(이미 K민감도 낮음을 확인함)

results = []
n_done = 0
CHECKPOINT = 300
for keys in combos:
    grp = df.groupby(keys)["control_success"]
    cum_n = grp.cumcount()
    cum_s = grp.cumsum() - df["control_success"]
    for k in K_VALUES:
        rate = (cum_s + k * g) / (cum_n + k)
        z_va = rate.to_numpy()[va_m]
        gain, pc = partial_gain(y_all[va_m], p_base, z_va)
        results.append(dict(keys="+".join(keys), n_keys=len(keys), k=k, gain=gain, partial_corr=pc,
                            mean_n=float(cum_n[va_m].mean())))
    n_done += 1
    if n_done % CHECKPOINT == 0:
        log(f"  {n_done}/{len(combos)} 완료...")

res = pd.DataFrame(results).sort_values("gain", ascending=False)
res.to_csv("feature_factory_massive_results.csv", index=False)
log(f"총 {len(res)}개 후보 스캔 완료")

# Bonferroni 보정 임계값 계산 (참조피처 12개 기준 ref_mean=-3.05, ref_sd=6.07 재사용)
ref_mean, ref_sd = -3.05, 6.07
n_cand = len(res)
alpha = 0.05
z_corrected = stats.norm.ppf(1 - alpha / (2 * n_cand))
threshold_corrected = ref_mean + z_corrected * ref_sd
threshold_naive = ref_mean + 2 * ref_sd

print()
print("=" * 100)
print(f"후보 {n_cand}개 스캔 완료")
print(f"미보정 임계값(phase93, 12개기준) = {threshold_naive:.2f}")
print(f"Bonferroni 보정 임계값({n_cand}개기준, z={z_corrected:.2f}) = {threshold_corrected:.2f}")
print()
print("상위 30개:")
print(res.head(30).to_string(index=False))
print()
survivors_naive = res[res.gain > threshold_naive]
survivors_corrected = res[res.gain > threshold_corrected]
print(f"미보정 기준 통과: {len(survivors_naive)}개")
print(f"Bonferroni보정 통과: {len(survivors_corrected)}개")
if len(survivors_corrected):
    print(survivors_corrected.to_string(index=False))
log(f"총 {time.time()-t0:.0f}s")
