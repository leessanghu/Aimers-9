"""phase92 — 기각한 피처들에 '비선형 정보'가 남아있는지 재검사.

문제의식: 지금까지 기각 판정에 쓴 partial_gain은 '선형' 부분상관이다.
피처가 U자형이거나 특정 임계값 근처에서만 의미 있으면 선형 상관은 0이어도
정보는 존재할 수 있다. phase64b 문서에도 "비선형성은 상위 피처만 구간더미로 확인"
이라고 적혀 있고, 실제로 기각한 피처들은 비선형 검사를 한 번도 하지 않았다.

질문을 두 개로 분리한다:
    (a) 이 피처에 정보가 있긴 한가        -> 모델 무관. 비모수(구간더미)로 측정 가능
    (b) 우리 모델이 그걸 쓸 수 있나        -> 모델 의존. RF/XGB가 다를 수 있음
(a)가 아니면 (b)는 무의미하다. 여기서는 (a)만 판정한다.

방법:
    선형 gain  = 1e5 * corr(ry, rz)^2                     (기존 partial_gain)
    비선형 gain = 1e5 * [R2(ry ~ z구간더미) - 순열귀무평균]  (12분위 구간)
    순열귀무는 z를 셔플해서 같은 계산 -> 자유도 편향((k-1)/n)을 실측으로 제거

판정:
    비선형 >> 선형  -> 선형 스크리너가 놓친 정보 있음. 모델 축(RF/XGB) 실험 의미 있음
    비선형 ~= 선형  -> 애초에 정보가 없음. 어떤 모델을 써도 못 뽑음

참조 피처(모델이 이미 강하게 쓰는 것)를 넣어 스크리너 자체를 검증한다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

VALID_SEASON = 2024
t0 = time.time()
rng = np.random.RandomState(42)


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def _clean(z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    return z


def residualize(y, p):
    """선형 잔차 (기존 partial_gain과 동일 기준)."""
    A = np.column_stack([np.ones(len(y)), p])
    return y - A @ np.linalg.lstsq(A, y, rcond=None)[0]


def residualize_np(y, p, n_bins=60):
    """비모수 잔차: p를 분위구간으로 나눠 구간평균을 제거한다.
    선형 잔차는 E[y|p]의 비선형(캘리브레이션 곡선의 휨)을 남기기 때문에,
    p와 상관된 아무 피처나 그 휨을 주워담아 가짜 비선형 신호로 보인다.
    (참조 피처가 비선형 gain 10점을 보인 것이 이 아티팩트였다)"""
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1])
    codes = np.searchsorted(qs, p).astype(np.int64)
    sums = np.bincount(codes, weights=y, minlength=n_bins)
    cnts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    means = np.divide(sums, cnts, out=np.zeros(n_bins), where=cnts > 0)
    return y - means[codes]


def linear_gain(ry, p, z):
    z = _clean(z)
    if z.std() == 0:
        return 0.0
    A = np.column_stack([np.ones(len(p)), p])
    rz = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    if rz.std() == 0:
        return 0.0
    return 1e5 * float(np.corrcoef(ry, rz)[0, 1]) ** 2


def _bin_r2(ry, codes, n_bins):
    """구간별 평균으로 ry를 설명했을 때의 R2 (설명분산/전체분산)."""
    sums = np.bincount(codes, weights=ry, minlength=n_bins)
    cnts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    ok = cnts > 0
    means = np.zeros(n_bins)
    means[ok] = sums[ok] / cnts[ok]
    explained = float(np.sum(cnts[ok] * means[ok] ** 2)) / len(ry)
    return explained / ry.var()


def nonlinear_gain(ry, z, n_bins=12, n_perm=8):
    z = _clean(z)
    if z.std() == 0:
        return 0.0, 0.0
    qs = np.quantile(z, np.linspace(0, 1, n_bins + 1)[1:-1])
    codes = np.searchsorted(qs, z).astype(np.int64)
    obs = _bin_r2(ry, codes, n_bins)
    nulls = []
    for i in range(n_perm):
        perm = rng.permutation(len(ry))
        nulls.append(_bin_r2(ry[perm], codes, n_bins))
    null_mean = float(np.mean(nulls))
    return 1e5 * (obs - null_mean), 1e5 * null_mean


log("데이터 로드...")
COLS = ["row_id", "season", "game_type", "pitcher_id", "batter_id", "control_success",
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_batter_n", "asof_batter_success_rate", "balls_before", "strikes_before", "inning"]
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig", usecols=COLS)
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)

z = np.load("phase67_cache/phase69_preds.npz")
p_gbdt = 0.5 * z["hgb"] + 0.5 * z["cat3"]
y_va = z["y"].astype(np.float64)
va = (df["season"] == VALID_SEASON).to_numpy()
assert va.sum() == len(y_va)
ry = residualize(y_va, p_gbdt)          # 선형 잔차 (선형 gain용, 기존 기준 유지)
ry_np = residualize_np(y_va, p_gbdt)    # 비모수 잔차 (비선형 gain용, p의 모든 함수와 직교)
log(f"valid={va.sum():,}  잔차 준비 완료 (선형/비모수 2종)")

# ---------------- 후보 구성 ----------------
cand = {}
dv = df[va].reset_index(drop=True)

# [참조] 모델이 이미 강하게 쓰는 것 -> 선형/비선형 둘 다 ~0 이어야 정상
cand["_ref_ball_rate"] = dv["asof_pitcher_ball_rate"].to_numpy(np.float64)
cand["_ref_success_rate"] = dv["asof_pitcher_success_rate"].to_numpy(np.float64)
cand["_ref_pitcher_n"] = dv["asof_pitcher_n"].fillna(0).to_numpy(np.float64)

# [의도축 - phase77에서 기각] 실패 모드 구성비
S = dv["asof_pitcher_success_rate"].to_numpy(np.float64)
R = dv["asof_pitcher_reverse_rate"].to_numpy(np.float64)
M = dv["asof_pitcher_middle_rate"].to_numpy(np.float64)
B = dv["asof_pitcher_ball_rate"].to_numpy(np.float64)
K = dv["asof_pitcher_strike_rate"].to_numpy(np.float64)
fail = np.clip(1.0 - S, 1e-6, None)
cand["rev_share"] = R / fail
cand["mid_share"] = M / fail
cand["safe_share"] = 1.0 - (R + M) / fail
cand["inplay_rate"] = 1.0 - B - K
cand["chase_intent"] = B - fail
cand["zone_minus_success"] = K - S

# [시대보정 - phase87/88에서 기각] 커리어레이트의 시대 오염
league = df.groupby("season")["control_success"].mean().to_dict()
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)


def cnt(col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)


S_, R_, M_ = [cnt(c) for c in ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                                "asof_pitcher_middle_rate"]]
ordr = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = df["pitcher_id"].to_numpy()[ordr]
n_o = n_[ordr]
step = np.zeros(len(df), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
sd_ = np.zeros(len(df))
sd_[ordr[:-1]] = np.diff(S_[ordr])
lab = pd.DataFrame({"pid": df["pitcher_id"].to_numpy()[step], "season": df["season"].to_numpy()[step],
                    "s": sd_[step], "n": 1.0})
per = lab.groupby(["pid", "season"]).agg(s=("s", "sum"), n=("n", "sum")).reset_index()
per["lg"] = per["season"].map(league)
per["resid"] = per["s"] - per["lg"] * per["n"]
hist = per[per.season <= 2023].groupby("pid").agg(n_tot=("n", "sum"), resid=("resid", "sum"),
                                                   s_tot=("s", "sum"))
skill = (hist["resid"] / hist["n_tot"]).rename("era_skill")
naive = (hist["s_tot"] / hist["n_tot"]).rename("naive_rate")
tmp = dv[["pitcher_id"]].join(skill, on="pitcher_id").join(naive, on="pitcher_id")
cand["era_skill"] = tmp["era_skill"].fillna(0.0).to_numpy(np.float64)
cand["era_correction"] = (tmp["era_skill"] + league[2024] - tmp["naive_rate"]).fillna(0.0).to_numpy(np.float64)

# [투수x타자 상호작용 - 오라클 1.6점으로 기각]
h = df[df.season <= 2023]
gl = h["control_success"].mean()
pr = h.groupby("pitcher_id")["control_success"].agg(ps="sum", pn="count")
br = h.groupby("batter_id")["control_success"].agg(bs="sum", bn="count")
pair = h.groupby(["pitcher_id", "batter_id"])["control_success"].agg(s="sum", n="count")
t2 = dv[["pitcher_id", "batter_id"]].join(pr, on="pitcher_id").join(br, on="batter_id") \
        .join(pair, on=["pitcher_id", "batter_id"]).fillna({"ps": 0, "pn": 0, "bs": 0, "bn": 0, "s": 0, "n": 0})
KP = 1000.0
p_rate = (t2["ps"] + KP * gl) / (t2["pn"] + KP)
b_rate = (t2["bs"] + KP * gl) / (t2["bn"] + KP)
pri = np.clip(p_rate + b_rate - gl, 0.02, 0.98).to_numpy(float)
for KK in [50.0, 500.0]:
    rate = (t2["s"].to_numpy(float) + KK * pri) / (t2["n"].to_numpy(float) + KK)
    cand[f"pair_diff_K{int(KK)}"] = rate - pri

# [상황 원시값 - 트리가 이미 쓰지만 비선형 여지 확인용]
cand["count_state"] = (dv["balls_before"] * 4 + dv["strikes_before"]).to_numpy(np.float64)
cand["inning"] = dv["inning"].to_numpy(np.float64)
cand["batter_n"] = dv["asof_batter_n"].fillna(0).to_numpy(np.float64)

log(f"후보 {len(cand)}개 구성 완료. 스크리닝 시작...")
rows = []
for name, v in cand.items():
    lg_ = linear_gain(ry, p_gbdt, v)
    nl, null = nonlinear_gain(ry_np, v)
    rows.append(dict(feature=name, linear=lg_, nonlinear=nl, null_bias=null,
                     ratio=(nl / lg_ if lg_ > 0.01 else np.nan)))
res = pd.DataFrame(rows).sort_values("nonlinear", ascending=False)

print()
print("=" * 78)
print(f"{'피처':<24}{'선형gain':>11}{'비선형gain':>12}{'비율':>8}{'귀무편향':>10}")
print("-" * 78)
for _, r in res.iterrows():
    ratio = f"{r['ratio']:.1f}x" if np.isfinite(r["ratio"]) else "-"
    mark = "  <- 참조" if r.feature.startswith("_ref") else ""
    print(f"{r.feature:<24}{r.linear:11.2f}{r.nonlinear:12.2f}{ratio:>8}{r.null_bias:10.2f}{mark}")

print()
print("해석 기준: 비선형 >> 선형 이면 선형 스크리너가 놓친 정보가 있다는 뜻.")
print("           둘 다 ~0 이면 애초에 정보가 없어 어떤 모델로도 못 뽑는다.")
big = res[(~res.feature.str.startswith("_ref")) & (res.nonlinear > 10)]
if len(big):
    print(f"\n비선형 gain > 10점인 기각피처: {list(big.feature)} -> 모델축(RF/XGB) 실험 가치 있음")
else:
    print("\n비선형 gain > 10점인 기각피처 없음 -> 정보 부재. 모델을 바꿔도 소용없음.")
res.to_csv("phase92_nonlinear_screen.csv", index=False)
log(f"총 {time.time()-t0:.0f}s")
