"""증분 잠재력 스크리너 — '재학습 delta' 대신 '천장을 얼마나 올리는가'로 피처를 판정한다.

왜 바꾸나 (phase63 결과):
  재학습 delta는 신호 + 보정 + 시드노이즈가 섞여 SD가 7~24였다. 그래서 embMLP처럼 실제로는
  358점짜리 신호를 가진 모델이 14.7점으로 찍혀 기각되는 일이 벌어졌다.

무엇을 재나:
  현재 최고 모델(GBDT v25)의 2024 예측 p를 고정 control로 넣고, 후보 피처 Z를 추가했을 때
  y ~ 1 + p + f(Z) 의 R^2가 얼마나 오르는지 본다. f는 분위수 구간 더미라 임의의 비선형/비단조
  관계도 잡는다. 증분 R^2 x 1e5 = '이 피처를 완벽히 활용했을 때 얻을 수 있는 최대 점수'.

편향 제거:
  단순 in-sample R^2는 파라미터 k개당 k/n 만큼 자동으로 오른다(n=253k, k=150이면 ~59점의 가짜 이득).
  그래서 split-half로 절반에서 계수를 적합하고 나머지 절반에서 R^2를 재는 방식으로 교차 평가한다.
  귀무가설 하에서 기대값이 0이 되므로 양수 = 진짜 신호다.

후보:
  [신규] trackman 물리 프로파일 / 폼 피처 / 역할 피처
  [기각됐던 것 재평가] career_volatility, arsenal_entropy, hidden_denominator
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from arsenal_entropy import K_ARSENAL, transform_arsenal
from career_volatility import K_VOL, build_volatility_table, transform_volatility
from formfeat import build_role_table, transform_form, transform_role
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from trackman_profile import build_trackman_profile, transform_trackman

VALID_SEASON = 2024
CACHE = "phase61_cache"
TM_CACHE = "phase64_trackman_profile.parquet"

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


# ======================================================================
# 스크리너 본체
# ======================================================================

def _bin_dummies(z, n_bins=12):
    """분위수 구간 원-핫 (첫 구간 drop). 임의 형태의 관계를 잡되 파라미터는 n_bins-1개로 제한."""
    z = np.asarray(z, dtype=np.float64)
    z = np.nan_to_num(z, nan=np.nanmedian(z) if np.isfinite(z).any() else 0.0)
    if np.nanstd(z) == 0:
        return np.zeros((len(z), 0))
    qs = np.unique(np.quantile(z, np.linspace(0, 1, n_bins + 1)))
    if len(qs) < 3:
        return np.zeros((len(z), 0))
    b = np.clip(np.digitize(z, qs[1:-1]), 0, len(qs) - 2)
    n_lev = int(b.max()) + 1
    D = np.zeros((len(z), n_lev - 1))
    for j in range(1, n_lev):
        D[:, j - 1] = (b == j)
    return D


def _r2_oos(y, X, fit_idx, ev_idx):
    coef, *_ = np.linalg.lstsq(X[fit_idx], y[fit_idx], rcond=None)
    pred = X[ev_idx] @ coef
    resid = y[ev_idx] - pred
    return 1 - resid.var() / y[ev_idx].var()


def incremental_potential(y, p_base, Z, n_bins=12, seed=42):
    """Z 추가 시 증분 잠재력(점수 단위). split-half 교차 평가로 비편향."""
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    rng = np.random.RandomState(seed)
    half = rng.rand(n) < 0.5
    A, B = np.where(half)[0], np.where(~half)[0]

    X0 = np.column_stack([np.ones(n), np.asarray(p_base, dtype=np.float64)])
    cols = [X0]
    Z = np.asarray(Z, dtype=np.float64)
    if Z.ndim == 1:
        Z = Z[:, None]
    for j in range(Z.shape[1]):
        D = _bin_dummies(Z[:, j], n_bins)
        if D.shape[1]:
            cols.append(D)
    X1 = np.column_stack(cols)

    gains = []
    for fit, ev in ((A, B), (B, A)):
        r0 = _r2_oos(y, X0, fit, ev)
        r1 = _r2_oos(y, X1, fit, ev)
        gains.append(r1 - r0)
    return 1e5 * float(np.mean(gains)), X1.shape[1] - 2


# ======================================================================
# 데이터 준비
# ======================================================================

log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())

va_mask = df["season"] == VALID_SEASON
va_i = df.index[va_mask]
df_va = df.loc[va_i]

# 2024 폴드 y / GBDT 예측 (phase61 캐시, 학습 train<=2023)
import glob
yfile = None
for pat in ("../phase3_tabm_preds/fold_2024_pred_*.csv", "phase3_preds/fold_2024_pred_*.csv"):
    m = sorted(glob.glob(pat))
    if m:
        yfile = m[0]
        break
y_va = pd.read_csv(yfile)["y_valid"].to_numpy(np.float64)
p_gbdt = np.load(f"{CACHE}/gbdt_v25_valid_pred.npy")
assert len(y_va) == len(p_gbdt) == len(df_va), (len(y_va), len(p_gbdt), len(df_va))
r = y_va.mean()
base_pot = 1e5 * (np.corrcoef(p_gbdt, y_va)[0, 1] ** 2)
log(f"2024 폴드 n={len(y_va):,}  GBDT 단독 잠재력={base_pot:.1f}")

# 공통: in-season 베이스라인 (폼 피처의 기준점)
log("in-season 베이스라인 계산...")
se = build_season_end_table(df)
dins_va = transform_inseason(df_va, se, g, sr)
base_success = dins_va["inseason_success_smooth"].to_numpy(np.float64)
# middle 베이스라인: 전역 middle 사전확률 (inseason에는 middle이 없어 근사)
gmid = float(df["asof_pitcher_middle_rate"].mean(skipna=True))
base_middle = np.full(len(df_va), gmid)

blocks = {}

# ---- [신규] 역할 ----
log("역할 프로파일 (선발/불펜)...")
role_tbl = build_role_table(df)
X_role = transform_role(df_va, role_tbl, sr)
blocks["역할 (선발/불펜) 6"] = X_role

# ---- [신규] 폼 ----
log("폼 피처...")
X_form = transform_form(df_va, X_role, base_success, base_middle)
blocks["폼 (자기베이스라인 대비) 11"] = X_form

# ---- [신규] trackman 물리 ----
log("trackman 물리 프로파일 (최초 1회는 오래 걸림)...")
if os.path.exists(TM_CACHE):
    prof = pd.read_parquet(TM_CACHE)
    log("  캐시 로드")
else:
    prof = build_trackman_profile()
    prof.to_parquet(TM_CACHE)
    log("  캐시 저장")
X_tm = transform_trackman(df_va, prof, sr)
log(f"  trackman 매칭율(2024) = {100*X_tm['tm_matched'].mean():.1f}%")
blocks["trackman 물리 17"] = X_tm

# ---- [기각됐던 것 재평가] ----
log("기각됐던 블록 재구성...")
vol_tbl = build_volatility_table(se)
blocks["[기각] career_volatility 5"] = transform_volatility(df_va, vol_tbl, sr, k=K_VOL)

amix = {c: float(df[c].mean(skipna=True)) for c in
        ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]}
blocks["[기각] arsenal_entropy 2"] = transform_arsenal(df_va, global_mix=amix, k=K_ARSENAL)


def infer_min_denominator(s_rate, m_rate, max_q, chunk=4000):
    q = np.arange(1, max_q + 1, dtype=np.float64)
    s_all = pd.Series(s_rate).to_numpy(np.float64)
    m_all = pd.Series(m_rate).to_numpy(np.float64)
    inf = np.ones(len(s_all))
    for st in range(0, len(s_all), chunk):
        s = s_all[st:st + chunk, None]; m = m_all[st:st + chunk, None]
        miss = np.isnan(s[:, 0]) | np.isnan(m[:, 0])
        s = np.nan_to_num(s); m = np.nan_to_num(m)
        err = np.maximum(np.abs(s * q - np.rint(s * q)), np.abs(m * q - np.rint(m * q))) / q
        ok = err <= 5.1e-7
        v = np.where(ok.any(1), ok.argmax(1) + 1, err.argmin(1) + 1)
        v[miss] = 1.0
        inf[st:st + len(v)] = v
    return inf


hid = pd.DataFrame(index=df_va.index)
for k, mq in ((1, 160), (3, 480), (5, 800)):
    hid[f"prev{k}_hidden_total_n"] = infer_min_denominator(
        df_va[f"asof_pitcher_prev{k}_game_success_rate"], df_va[f"asof_pitcher_prev{k}_game_middle_rate"], mq)
hid["prev3_hidden_avg_n"] = hid["prev3_hidden_total_n"] / 3
hid["prev5_hidden_avg_n"] = hid["prev5_hidden_total_n"] / 5
hid["prev1_vs_prev3_workload"] = hid["prev1_hidden_total_n"] - hid["prev3_hidden_avg_n"]
hid["prev3_vs_prev5_workload"] = hid["prev3_hidden_avg_n"] - hid["prev5_hidden_avg_n"]
blocks["[기각] hidden_denominator 7"] = hid.astype(np.float64)


# ======================================================================
# 스크리닝
# ======================================================================

log("\n" + "=" * 88)
log("블록별 증분 잠재력 (GBDT v25 위에 얹었을 때 천장 상승분, split-half 비편향)")
log("=" * 88)
print(f"{'블록':<34}{'증분점수':>12}{'파라미터':>10}")
print("-" * 56)
block_res = {}
for name, X in blocks.items():
    gain, k = incremental_potential(y_va, p_gbdt, X.to_numpy(np.float64))
    block_res[name] = gain
    print(f"{name:<34}{gain:12.1f}{k:10d}")

log("\n" + "=" * 88)
log("개별 피처 증분 잠재력 (상위)")
log("=" * 88)
rows = []
for name, X in blocks.items():
    for c in X.columns:
        gain, _ = incremental_potential(y_va, p_gbdt, X[c].to_numpy(np.float64))
        rows.append((gain, c, name))
rows.sort(reverse=True)
print(f"{'피처':<28}{'증분점수':>11}   블록")
print("-" * 76)
for gain, c, name in rows[:28]:
    print(f"{c:<28}{gain:11.1f}   {name}")

log("\n" + "=" * 88)
log("신규 블록 합동 (역할+폼+trackman)")
log("=" * 88)
X_all = pd.concat([blocks["역할 (선발/불펜) 6"], blocks["폼 (자기베이스라인 대비) 11"],
                   blocks["trackman 물리 17"]], axis=1)
gain_all, k_all = incremental_potential(y_va, p_gbdt, X_all.to_numpy(np.float64))
print(f"  합동 증분 = {gain_all:.1f}  (파라미터 {k_all})")
print(f"  개별 합산 = {sum(block_res[n] for n in ['역할 (선발/불펜) 6','폼 (자기베이스라인 대비) 11','trackman 물리 17']):.1f}"
      f"  -> 상관 때문에 합동이 더 작은 게 정상")
print()
print(f"  GBDT 단독 잠재력      {base_pot:8.1f}")
print(f"  + 신규 블록 전부      {base_pot+gain_all:8.1f}")
print(f"  (참고) 현재 실제 점수 {max(0,1e5*(1-np.mean((p_gbdt-y_va)**2)/(r*(1-r)))):8.1f}")

os.makedirs(CACHE, exist_ok=True)
pd.DataFrame([{"block": k, "gain": v} for k, v in block_res.items()]).to_csv(
    f"{CACHE}/phase64_block_gains.csv", index=False)
pd.DataFrame([{"feature": c, "block": n, "gain": gnum} for gnum, c, n in rows]).to_csv(
    f"{CACHE}/phase64_feature_gains.csv", index=False)
log("저장 완료")
