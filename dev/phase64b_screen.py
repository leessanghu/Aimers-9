"""증분 잠재력 스크리너 v2 — 부분상관 기반 (phase64의 검정력 문제 수정).

phase64의 결함: 피처당 12구간 더미(파라미터 11개)를 split-half OOS로 평가했는데,
귀무가설 하에서 OOS 이득의 기대값은 0이 아니라 -k/n_fit 이다(노이즈 계수를 적합해서
다른 절반에 적용하므로 반드시 손해). k=149면 -117점이 '정상'이라 진짜 신호가 다 묻혔다.

v2 방식:
  개별 피처 -> p를 통제한 부분상관 (자유도 1). 귀무편향 = 1/n = 0.4점. 검정력 300배.
      증분점수 ~= 1e5 * partial_corr(z, y | p)^2
      +100점짜리 피처면 partial_corr = 0.032 = 16 시그마. 확실히 잡힌다.
  블록      -> 선형 다변량 in-sample R^2에 자유도 보정 (-k/n). k가 작아 편향도 작다.
  비선형성  -> 상위 피처만 구간더미 + 순열귀무로 따로 확인.

해석 기준 (n=253,507):
  partial_corr의 표준오차 = 1/sqrt(n) = 0.00199
  즉 증분점수 4점 = 1시그마, 16점 = 2시그마, 36점 = 3시그마
"""
import glob
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from arsenal_entropy import K_ARSENAL, transform_arsenal
from career_volatility import K_VOL, build_volatility_table, transform_volatility
from formfeat import build_role_table, transform_form, transform_role
from inseason import build_season_end_table, transform_inseason
from trackman_profile import build_trackman_profile, transform_trackman

VALID_SEASON = 2024
CACHE = "phase61_cache"
TM_CACHE = "phase64_trackman_profile.parquet"
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


def _clean(z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    return z


def partial_gain(y, p, z):
    """p를 통제한 z의 증분 잠재력 (점수 단위). 자유도 1이라 귀무편향 ~0.4점."""
    z = _clean(z)
    if z.std() == 0:
        return 0.0, 0.0
    # p로 각각 회귀한 잔차
    A = np.column_stack([np.ones(len(y)), p])
    cy = np.linalg.lstsq(A, y, rcond=None)[0]
    cz = np.linalg.lstsq(A, z, rcond=None)[0]
    ry, rz = y - A @ cy, z - A @ cz
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


def block_gain_linear(y, p, Z):
    """블록 전체를 선형으로 추가했을 때 증분 R^2 (자유도 보정)."""
    Z = np.column_stack([_clean(Z[:, j]) for j in range(Z.shape[1])])
    keep = [j for j in range(Z.shape[1]) if Z[:, j].std() > 0]
    Z = Z[:, keep]
    n, k = len(y), Z.shape[1]
    X0 = np.column_stack([np.ones(n), p])
    X1 = np.column_stack([X0, Z])
    def r2(X):
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - (y - X @ c).var() / y.var()
    raw = r2(X1) - r2(X0)
    return 1e5 * (raw - k / n), k


def block_gain_binned(y, p, Z, n_bins=8, n_perm=3, seed=42):
    """비선형 포함 블록 증분 — 순열귀무로 편향을 실측 보정."""
    def bins(z):
        z = _clean(z)
        if z.std() == 0:
            return np.zeros((len(z), 0))
        qs = np.unique(np.quantile(z, np.linspace(0, 1, n_bins + 1)))
        if len(qs) < 3:
            return np.zeros((len(z), 0))
        b = np.clip(np.digitize(z, qs[1:-1]), 0, len(qs) - 2)
        nl = int(b.max()) + 1
        return np.column_stack([(b == j).astype(np.float64) for j in range(1, nl)])

    n = len(y)
    X0 = np.column_stack([np.ones(n), p])

    def gain_for(Zm):
        cols = [X0] + [bins(Zm[:, j]) for j in range(Zm.shape[1])]
        cols = [c for c in cols if c.shape[1] > 0]
        X1 = np.column_stack(cols)
        def r2(X):
            c = np.linalg.lstsq(X, y, rcond=None)[0]
            return 1 - (y - X @ c).var() / y.var()
        return r2(X1) - r2(X0), X1.shape[1] - 2

    obs, k = gain_for(Z)
    nulls = []
    for i in range(n_perm):
        pr = np.random.RandomState(1000 + i).permutation(n)
        nulls.append(gain_for(Z[pr])[0])
    return 1e5 * (obs - float(np.mean(nulls))), k


# ======================================================================
log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())
df_va = df[df["season"] == VALID_SEASON]

yfile = (sorted(glob.glob("../phase3_tabm_preds/fold_2024_pred_*.csv")) +
         sorted(glob.glob("phase3_preds/fold_2024_pred_*.csv")))[0]
y_va = pd.read_csv(yfile)["y_valid"].to_numpy(np.float64)
p_gbdt = np.load(f"{CACHE}/gbdt_v25_valid_pred.npy")
r = y_va.mean()
base_pot = 1e5 * np.corrcoef(p_gbdt, y_va)[0, 1] ** 2
log(f"n={len(y_va):,}  GBDT 잠재력={base_pot:.1f}  (1시그마 = 4.0점)")

se = build_season_end_table(df)
dins_va = transform_inseason(df_va, se, g, sr)
base_success = dins_va["inseason_success_smooth"].to_numpy(np.float64)
gmid = float(df["asof_pitcher_middle_rate"].mean(skipna=True))
base_middle = np.full(len(df_va), gmid)

blocks = {}
log("역할/폼...")
role_tbl = build_role_table(df)
X_role = transform_role(df_va, role_tbl, sr)
blocks["역할(선발/불펜)"] = X_role
blocks["폼"] = transform_form(df_va, X_role, base_success, base_middle)

log("trackman...")
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
if not os.path.exists(TM_CACHE):
    prof.to_parquet(TM_CACHE)
X_tm = transform_trackman(df_va, prof, sr)
blocks["trackman물리"] = X_tm
log(f"  매칭율 {100*X_tm['tm_matched'].mean():.1f}%")

vol_tbl = build_volatility_table(se)
blocks["[기각]volatility"] = transform_volatility(df_va, vol_tbl, sr, k=K_VOL)
amix = {c: float(df[c].mean(skipna=True)) for c in
        ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]}
blocks["[기각]arsenal"] = transform_arsenal(df_va, global_mix=amix, k=K_ARSENAL)

# 참고용: 이미 모델에 들어있는 강한 피처의 증분 (0에 가까워야 정상 — 스크리너 검증)
blocks["[검증]기존피처"] = pd.DataFrame({
    "inseason_success_smooth": base_success,
    "asof_pitcher_success_rate": df_va["asof_pitcher_success_rate"].to_numpy(np.float64),
}, index=df_va.index)

# ======================================================================
log("\n" + "=" * 78)
log("개별 피처 증분 잠재력 — 부분상관 (1시그마=4.0점, 2시그마=16점, 3시그마=36점)")
log("=" * 78)
rows = []
for name, X in blocks.items():
    for c in X.columns:
        gain, pc = partial_gain(y_va, p_gbdt, X[c].to_numpy())
        rows.append((gain, pc, c, name))
rows.sort(reverse=True)
print(f"{'피처':<30}{'증분점수':>10}{'부분상관':>10}{'시그마':>8}   블록")
print("-" * 84)
for gain, pc, c, name in rows:
    if abs(gain) < 1.0 and name != "[검증]기존피처":
        continue
    print(f"{c:<30}{gain:10.1f}{pc:+10.4f}{abs(pc)/0.00199:8.1f}   {name}")

log("\n" + "=" * 78)
log("블록 증분 잠재력")
log("=" * 78)
print(f"{'블록':<24}{'선형(df보정)':>14}{'구간더미(순열보정)':>20}{'피처수':>8}")
print("-" * 66)
for name, X in blocks.items():
    Z = X.to_numpy(np.float64)
    gl, k = block_gain_linear(y_va, p_gbdt, Z)
    gb, kb = block_gain_binned(y_va, p_gbdt, Z)
    print(f"{name:<24}{gl:14.1f}{gb:20.1f}{X.shape[1]:8d}")

log("\n" + "=" * 78)
log("신규 3블록 합동")
log("=" * 78)
X_all = pd.concat([blocks["역할(선발/불펜)"], blocks["폼"], blocks["trackman물리"]], axis=1)
gl, k = block_gain_linear(y_va, p_gbdt, X_all.to_numpy(np.float64))
gb, kb = block_gain_binned(y_va, p_gbdt, X_all.to_numpy(np.float64))
print(f"  선형(df보정)       {gl:8.1f}   (피처 {X_all.shape[1]})")
print(f"  구간더미(순열보정) {gb:8.1f}")
print()
print(f"  GBDT 단독 잠재력   {base_pot:8.1f}")
print(f"  + 신규 전부        {base_pot+max(gl,gb):8.1f}")
print(f"  필요 (LB 1500 환산) {1500*861.1/981.44:8.1f}")

pd.DataFrame([{"feature": c, "block": n, "gain": gn, "partial_corr": pc}
              for gn, pc, c, n in rows]).to_csv(f"{CACHE}/phase64b_gains.csv", index=False)
log("저장 완료")
