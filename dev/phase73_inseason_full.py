"""in-season middle/strike 후보를 증분 잠재력 + 기존 강피처와의 magnitude 비교로 검증.

목표: '목표점수 역산'이 아니라 'in-season과 같은 클래스(직접 결과 x 당해시즌 x 큰 표본)'의
피처를 찾는 것. 그래서 두 가지를 같이 본다.
  1) 증분 잠재력 (부분상관) — 이미 모델이 아는 것 위에 새로 더하는 정보량
  2) 기존 강피처와의 비교 — 같은 방식으로 만든 inseason_success/reverse가 몇 점 나오는지
     같이 재서, 새 후보가 그 클래스에 속하는지 상대 비교

주의: inseason_success_smooth는 이미 모델 안에 있으므로 증분은 0으로 나온다(phase64b에서 검증됨).
      대신 '모델에서 그 피처를 빼면 얼마나 잃는가'를 재면 그 피처의 진짜 크기를 알 수 있다.
      여기서는 후보군끼리의 상대 비교 + 잔차 대비 단독 설명력으로 판단한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from inseason_full import (build_global_priors, build_season_end_table_full,
                           transform_inseason_full)

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
    cy = np.linalg.lstsq(A, y, rcond=None)[0]
    cz = np.linalg.lstsq(A, z, rcond=None)[0]
    ry, rz = y - A @ cy, z - A @ cz
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


def solo_r2(y, z):
    """모델과 무관하게 z 단독이 y를 설명하는 양 (피처 자체의 '크기' 감각용)."""
    z = _clean(z)
    if z.std() == 0:
        return 0.0
    return 1e5 * float(np.corrcoef(z, y)[0, 1]) ** 2


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


log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())
d = df[df["season"] == VALID_SEASON].copy().reset_index(drop=True)

p_ctrl = np.load(f"{CACHE}/gbdt_v26_valid_pred.npy")
y = d["control_success"].to_numpy(np.float64)
assert len(p_ctrl) == len(y)
log(f"n={len(y):,}  control=GBDT(132피처) score={max(0,1e5*(1-np.mean((p_ctrl-y)**2)/(y.mean()*(1-y.mean())))):.1f}")
log(f"1시그마 = {1e5/len(y):.2f}점 (부분상관 {1/np.sqrt(len(y)):.5f})")

# ---- 기존 in-season 블록 (비교 기준) ----
log("기존 in-season 계산...")
se = build_season_end_table(df)
X_ins = transform_inseason(d, se, g, sr)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([d["pitcher_id"], d["season"] - 1])
n_end = np.nan_to_num(piv["N_end"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)

# ---- 신규 middle/strike ----
log("신규 in-season middle/strike 계산...")
tbl_full = build_season_end_table_full(df)
priors = build_global_priors(df)
X_new = transform_inseason_full(d, tbl_full, priors, sr, n_end,
                                inseason_success=X_ins["inseason_success_smooth"].to_numpy(),
                                inseason_reverse=X_ins["inseason_reverse_smooth"].to_numpy())
log(f"  전역 사전확률: middle={priors['middle']:.4f}  strike={priors['strike']:.4f}")
for c in X_new.columns:
    v = X_new[c]
    log(f"  {c:<30} mean={v.mean():+.4f} SD={v.std():.4f}")

log("\n" + "=" * 86)
log("단독 설명력 — 피처 '크기' 비교 (모델 무관, y와의 상관만)")
log("=" * 86)
print(f"{'피처':<34}{'단독설명력':>12}   비고")
print("-" * 74)
ref = [("inseason_success_smooth", X_ins["inseason_success_smooth"].to_numpy(), "기존 magnitude 1위 (0.030)"),
       ("inseason_reverse_smooth", X_ins["inseason_reverse_smooth"].to_numpy(), "기존 magnitude 2위 (0.022)"),
       ("inseason_ball_smooth", X_ins["inseason_ball_smooth"].to_numpy(), "기존"),
       ("asof_pitcher_success_rate", d["asof_pitcher_success_rate"].fillna(g).to_numpy(), "커리어(공식컬럼)")]
for nm, z, note in ref:
    print(f"{nm:<34}{solo_r2(y, z):12.1f}   {note}")
print("-" * 74)
for c in X_new.columns:
    print(f"{c:<34}{solo_r2(y, X_new[c].to_numpy()):12.1f}   [신규]")

log("\n" + "=" * 86)
log("증분 잠재력 — GBDT(132피처) 위에 새로 더하는 정보량")
log("=" * 86)
print(f"{'피처':<34}{'증분점수':>10}{'부분상관':>11}{'시그마':>9}")
print("-" * 66)
for nm, z, note in ref:
    gn, pc = partial_gain(y, p_ctrl, z)
    print(f"{nm:<34}{gn:10.2f}{pc:+11.4f}{abs(pc)*np.sqrt(len(y)):9.1f}   (이미 모델에 있음)")
print("-" * 66)
rows = []
for c in X_new.columns:
    gn, pc = partial_gain(y, p_ctrl, X_new[c].to_numpy())
    rows.append((gn, c, pc))
    print(f"{c:<34}{gn:10.2f}{pc:+11.4f}{abs(pc)*np.sqrt(len(y)):9.1f}   [신규]")

gb, k = block_gain(y, p_ctrl, X_new.to_numpy(np.float64))
print(f"\n  [신규 블록 합동] 증분 = {gb:.1f}  (피처 {X_new.shape[1]})")

log("\n" + "=" * 86)
log("판정")
log("=" * 86)
best = max(rows)
print(f"  최강 신규 피처: {best[1]}  증분 {best[0]:.1f}점 ({abs(best[2])*np.sqrt(len(y)):.1f}시그마)")
print(f"  비교: 지금까지 추가한 것들의 최강값")
print(f"        count_diff(v26)        9.3  (4.9시그마)")
print(f"        bat_inseason_smooth    17.1 (6.6시그마)")
print(f"        form5_middle            6.0 (3.9시그마)")
print(f"        tm_ivb_sd               1.9 (2.2시그마)")
print()
print(f"  실현율 보정(실측/로컬 = 0.6) 적용시 예상 실측 이득 = {gb*0.6:+.1f}점")

os.makedirs(CACHE, exist_ok=True)
pd.DataFrame([{"feature": c, "gain": gn, "pc": pc} for gn, c, pc in rows]).to_csv(
    f"{CACHE}/phase73_gains.csv", index=False)
log(f"\n총 {time.time()-t0:.0f}s")
