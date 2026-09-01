"""신규 정보원 탐색 + trackman 실패 원인 규명 + v26 보정 여지 실측.

[A] 보정 여지 재측정
    phase66에서 잰 b=1.39는 es_i(2024 후반부, 같은 시즌)라 신뢰 불가였다.
    여기서는 진짜 미지시즌 폴드(train<=2023 -> valid=2024)의 phase67 예측으로 다시 잰다.

[B] trackman이 왜 실패했나 — 가설: '결과를 이미 아는데 원인을 줘봐야 소용없다'
    물리량(릴리스 일관성 등)은 제구력의 원인이고, 우리는 결과(성공률 이력)를 직접 관측한다.
    -> 성공률 이력이 신뢰할 만한 고표본 투수에게는 물리량이 무용지물이어야 한다.
    -> 반대로 저표본 투수(asof_pitcher_n 하위)에게는 물리량이 살아나야 한다.
    asof_pitcher_n 4분위별로 trackman 증분을 따로 재서 이 가설을 검정한다.

[C] 신규 정보원 1 — TTO (Times Through the Order)
    투수는 같은 타순을 세 번째 상대할 때 급격히 나빠진다(야구에서 가장 견고한 효과 중 하나).
    행 내부 정보만으로 '이 투수 팀이 지금까지 상대한 타자 수'를 복원할 수 있다:
        batters_faced ~= 3*(inning-1) + outs_before + num_runners_on + runs_against
        (모든 타자는 아웃되거나, 출루해 있거나, 득점했다)
    이건 4개 변수의 합이라 트리가 근사하기 매우 비효율적인 형태다(crosses.py와 같은 철학).
    규칙 준수: 전부 그 행의 공식 컬럼만 사용. test 행 간 참조 없음.

[D] 신규 정보원 2 — 타자 쪽 조건부
    투수에 대해서는 inseason/lastyear/platoon/inning/count를 다 만들었는데 타자 쪽은
    공식 asof_batter_* 4개가 전부다. phase65 오라클: batter_id 천장 148 (pitcher 840 대비 작지만
    아직 안 짜낸 영역). 타자 in-season 폼을 투수와 동일한 차분 트릭으로 만들어 검정한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from formfeat import build_role_table, transform_role
from inseason import build_season_end_table
from metrics import evaluate
from trackman_profile import build_trackman_profile, transform_trackman

VALID_SEASON = 2024
CACHE = "phase67_cache"
TM_CACHE = "phase64_trackman_profile.parquet"
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
    """p를 통제한 z의 증분 잠재력(점수). 자유도 1이라 귀무편향 ~1e5/n."""
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


# ======================================================================
log("데이터 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
sr = sorted(df["season"].unique().tolist())
d = df[df["season"] == VALID_SEASON].copy().reset_index(drop=True)

p_gbdt = np.load(f"{CACHE}/gbdt_v26_valid_pred.npy")
y = d["control_success"].to_numpy(np.float64)
assert len(p_gbdt) == len(y), (len(p_gbdt), len(y))
r = y.mean()
bsref = r * (1 - r)
score = max(0, 1e5 * (1 - np.mean((p_gbdt - y) ** 2) / bsref))
sigma1 = 1e5 * (1 / len(y))   # 자유도1 귀무편향 = 1시그마 눈금
log(f"n={len(y):,}  GBDT(132피처) score={score:.1f}  (1시그마 = {1e5/len(y)*1:.2f}점 아님, 아래 참조)")

# ---------- [A] 보정 여지 ----------
log("\n" + "=" * 76)
log("[A] 보정 여지 — 진짜 미지시즌 폴드에서 재측정")
log("=" * 76)
bias = p_gbdt.mean() - r
cov = np.mean((p_gbdt - p_gbdt.mean()) * (y - r))
b_opt = cov / p_gbdt.var()
a_opt = r - b_opt * p_gbdt.mean()
p_cal = np.clip(a_opt + b_opt * p_gbdt, 1e-6, 1 - 1e-6)
score_cal = max(0, 1e5 * (1 - np.mean((p_cal - y) ** 2) / bsref))
rho = np.corrcoef(p_gbdt, y)[0, 1]
print(f"  현재 score          {score:8.1f}")
print(f"  잠재력(=1e5*rho^2)  {1e5*rho**2:8.1f}")
print(f"  보정후 score        {score_cal:8.1f}   (여지 {score_cal-score:+.1f})")
print(f"  bias={bias:+.5f}  b_opt={b_opt:.4f}   (phase66의 es_i 추정 b=1.39는 역시 과대였음)")
print(f"  -> '160점'은 폐기한 embMLP 얘기였고, 우리 GBDT의 실제 보정 여지는 위 값이다.")

# ---------- [B] trackman 저표본 가설 ----------
log("\n" + "=" * 76)
log("[B] trackman 실패 원인 — '고표본 투수에겐 무용, 저표본에겐 유효' 가설 검정")
log("=" * 76)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
X_tm = transform_trackman(d, prof, sr)
tm_cols = [c for c in X_tm.columns if c not in ("tm_matched",)]

n_pitcher = d["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
q = np.quantile(n_pitcher, [0.25, 0.5, 0.75])
strata = {
    "Q1 저표본 (n<=%.0f)" % q[0]: n_pitcher <= q[0],
    "Q2 (%.0f<n<=%.0f)" % (q[0], q[1]): (n_pitcher > q[0]) & (n_pitcher <= q[1]),
    "Q3 (%.0f<n<=%.0f)" % (q[1], q[2]): (n_pitcher > q[1]) & (n_pitcher <= q[2]),
    "Q4 고표본 (n>%.0f)" % q[2]: n_pitcher > q[2],
}
print(f"{'구간':<26}{'n':>9}{'trackman블록 증분':>18}{'1시그마':>9}")
print("-" * 64)
for name, m in strata.items():
    if m.sum() < 5000:
        continue
    gb, k = block_gain(y[m], p_gbdt[m], X_tm[tm_cols].to_numpy(np.float64)[m])
    s1 = 1e5 / m.sum()
    print(f"{name:<26}{m.sum():9,}{gb:18.1f}{s1:9.1f}")
print("  (1시그마는 구간별 표본수에 따라 달라짐. 증분이 1시그마의 2배 넘으면 유의)")

# ---------- [C] TTO ----------
log("\n" + "=" * 76)
log("[C] 신규 — TTO (Times Through the Order) / 추정 투구수")
log("=" * 76)
is_top = d["top_bottom"].astype(str).str.upper().str[0].eq("T").to_numpy()
runs_against = np.where(is_top, d["run_top_before"].to_numpy(np.float64),
                        d["run_bot_before"].to_numpy(np.float64))
# 규약 검증: 투수팀 기준 점수차 == 투수팀득점 - 상대득점
runs_for = np.where(is_top, d["run_bot_before"].to_numpy(np.float64),
                    d["run_top_before"].to_numpy(np.float64))
chk = np.mean(np.abs((runs_for - runs_against) - d["score_diff_pitcher_team"].to_numpy(np.float64)) < 1e-6)
print(f"  top_bottom 규약 검증: score_diff_pitcher_team 일치율 = {100*chk:.1f}%")
if chk < 0.9:
    runs_against, runs_for = runs_for, runs_against
    print("  -> 규약이 반대여서 교체함")

inning = d["inning"].to_numpy(np.float64)
outs = d["outs_before"].to_numpy(np.float64)
runners = d["num_runners_on"].to_numpy(np.float64)
bf = 3 * (inning - 1) + outs + runners + runs_against

role_tbl = build_role_table(df)
X_role = transform_role(d, role_tbl, sr)
ppa = np.clip(X_role["role_ppa"].to_numpy(np.float64), 1.0, None)
starter = X_role["role_first_inn_share"].to_numpy(np.float64)

X_tto = pd.DataFrame({
    "tto_batters_faced": bf,
    "tto_times_through": bf / 9.0,
    "tto_est_pitches": bf * 3.8,
    "tto_x_starter": bf * starter,              # 선발이면 그 타자들을 자기가 다 상대함
    "tto_vs_own_ppa": (bf * 3.8) / ppa,          # 자기 평소 등판량 대비 소진 비율
    "tto_third_time": np.clip(bf / 9.0 - 2.0, 0, None),  # 3순회 진입 이후만 켜지는 힌지
})
for c in X_tto.columns:
    g, pc = partial_gain(y, p_gbdt, X_tto[c].to_numpy())
    print(f"  {c:<22} 증분={g:7.2f}  부분상관={pc:+.4f}  시그마={abs(pc)*np.sqrt(len(y)):.1f}")
gb, k = block_gain(y, p_gbdt, X_tto.to_numpy(np.float64))
print(f"  [블록 합동] 증분={gb:.1f}  (피처 {X_tto.shape[1]})")

# ---------- [D] 타자 쪽 조건부 ----------
log("\n" + "=" * 76)
log("[D] 신규 — 타자 in-season 폼 (투수와 동일한 차분 트릭)")
log("=" * 76)
bt = (df.groupby(["batter_id", "season"])["control_success"]
        .agg(S="sum", N="count").sort_index())
bt = bt.groupby(level=0).cumsum().reset_index()
pv = {c: bt.pivot(index="batter_id", columns="season", values=c)
            .reindex(columns=sr).ffill(axis=1).stack(future_stack=True) for c in ("S", "N")}
idx = pd.MultiIndex.from_arrays([d["batter_id"], d["season"] - 1])
S_end = np.nan_to_num(pv["S"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
N_end = np.nan_to_num(pv["N"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)

n_now = d["asof_batter_n"].fillna(0).to_numpy(np.float64)
s_now = np.round(d["asof_batter_success_rate"].fillna(0).to_numpy(np.float64) * n_now)
n_seas = np.clip(n_now - N_end, 0, None)
s_seas = np.clip(s_now - S_end, 0, None)
gb_rate = float(df["control_success"].mean())
K_B = 30.0
X_bat = pd.DataFrame({
    "bat_inseason_smooth": (s_seas + K_B * gb_rate) / (n_seas + K_B),
    "bat_inseason_n": np.log1p(n_seas),
    "bat_ly_rate": np.divide(S_end, np.maximum(N_end, 1), out=np.full(len(d), gb_rate), where=N_end > 0),
    "bat_ly_n": np.log1p(N_end),
})
X_bat["bat_inseason_minus_career"] = X_bat["bat_inseason_smooth"] - d["asof_batter_success_rate"].fillna(gb_rate).to_numpy()
for c in X_bat.columns:
    g, pc = partial_gain(y, p_gbdt, X_bat[c].to_numpy())
    print(f"  {c:<26} 증분={g:7.2f}  부분상관={pc:+.4f}  시그마={abs(pc)*np.sqrt(len(y)):.1f}")
gb2, k2 = block_gain(y, p_gbdt, X_bat.to_numpy(np.float64))
print(f"  [블록 합동] 증분={gb2:.1f}  (피처 {X_bat.shape[1]})")

log("\n" + "=" * 76)
log("종합")
log("=" * 76)
print(f"  보정 여지            {score_cal-score:+8.1f}")
print(f"  TTO 블록             {gb:+8.1f}")
print(f"  타자 in-season 블록   {gb2:+8.1f}")
print(f"  (참고) 1시그마 눈금   {1e5/np.sqrt(len(y))**2*1:.2f}점 = 부분상관 {1/np.sqrt(len(y)):.5f}")
log(f"\n총 {time.time()-t0:.0f}s")
