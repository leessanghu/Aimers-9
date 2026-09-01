"""phase93 — 편향 없는 비모수 증분 스크리너 (분할반) + 참조피처 12개로 기준선 검증.

phase92의 문제: 순열귀무로 자유도 편향을 빼려 했으나, 참조피처(모델이 이미 강하게 쓰는 것)가
여전히 4.47~9.22점을 보였다. 즉 편향이 다 안 빠졌다. 원인은 두 가지였다.
  1) 선형 잔차는 E[y|p]의 휨을 남긴다 -> 비모수 잔차로 고쳤으나
  2) 순열귀무는 구간크기 불균형/과적합 편향을 완전히 제거하지 못한다.

이번 수정 — 분할반(split-half):
    구간평균을 절반(A)에서 구하고 나머지 절반(B)에서 평가한다.
    신호가 없으면 out-of-sample 설명분산의 기대값이 0(약간 음수)이므로,
    구간 개수와 무관하게 편향이 구조적으로 사라진다.
    5개 시드로 반복해 평균 + SD를 낸다.

검증: 참조피처 12개(모델이 이미 쓰는 것, 강/약/연속/이산 골고루)를 넣는다.
      이들이 0 근처로 나와야 스크리너를 믿을 수 있다.

후보: 지금까지 기각한 것 전부 (트릭맨 실행실패, 의도축, 시대보정, 투수x타자 쌍)
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

VALID_SEASON = 2024
TM_PATH = "../data/trackman_history.csv"
MAP_PATH = "pitcher_map.csv"
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def _clean(z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    return z


def residualize_np(y, p, n_bins=200):
    """p의 어떤 함수와도 직교하는 잔차 (구간평균 제거)."""
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1])
    codes = np.searchsorted(qs, p).astype(np.int64)
    sums = np.bincount(codes, weights=y, minlength=n_bins)
    cnts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    means = np.divide(sums, cnts, out=np.zeros(n_bins), where=cnts > 0)
    return y - means[codes]


def linear_gain(y, p, z):
    z = _clean(z)
    if z.std() == 0:
        return 0.0
    A = np.column_stack([np.ones(len(y)), p])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rz = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    if rz.std() == 0:
        return 0.0
    return 1e5 * float(np.corrcoef(ry, rz)[0, 1]) ** 2


def splithalf_gain(ry, z, n_bins=12, n_splits=5):
    """구간평균을 A에서 학습해 B에서 평가. 신호 없으면 기대값 ~0 (음수 가능)."""
    z = _clean(z)
    if z.std() == 0:
        return 0.0, 0.0
    qs = np.quantile(z, np.linspace(0, 1, n_bins + 1)[1:-1])
    qs = np.unique(qs)
    if len(qs) == 0:
        return 0.0, 0.0
    codes = np.searchsorted(qs, z).astype(np.int64)
    nb = codes.max() + 1
    vary = ry.var()
    gains = []
    for s in range(n_splits):
        rs = np.random.RandomState(1000 + s)
        half = rs.rand(len(ry)) < 0.5
        sums = np.bincount(codes[half], weights=ry[half], minlength=nb)
        cnts = np.bincount(codes[half], minlength=nb).astype(np.float64)
        gmean = ry[half].mean()
        means = np.where(cnts > 0, np.divide(sums, np.maximum(cnts, 1)), gmean)
        pred = means[codes[~half]]
        yb = ry[~half]
        mse = np.mean((yb - pred) ** 2)
        r2 = 1.0 - mse / yb.var()
        gains.append(1e5 * r2 * (yb.var() / vary))
    return float(np.mean(gains)), float(np.std(gains, ddof=1))


log("데이터 로드...")
COLS = ["row_id", "season", "game_type", "pitcher_id", "batter_id", "control_success",
        "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
        "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
        "balls_before", "strikes_before", "inning", "outs_before", "li"]
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig", usecols=COLS)
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)

zc = np.load("phase67_cache/phase69_preds.npz")
p_gbdt = 0.5 * zc["hgb"] + 0.5 * zc["cat3"]
y_va = zc["y"].astype(np.float64)
va = (df["season"] == VALID_SEASON).to_numpy()
assert va.sum() == len(y_va)
ry = residualize_np(y_va, p_gbdt, n_bins=200)
log(f"valid={va.sum():,}  비모수 잔차(200구간) 준비")

dv = df[va].reset_index(drop=True)
REF, CAND = {}, {}

# ---------------- 참조 피처 12개 (모델이 이미 쓰는 것) ----------------
REF["ref_success_rate"] = dv["asof_pitcher_success_rate"].to_numpy(np.float64)
REF["ref_ball_rate"] = dv["asof_pitcher_ball_rate"].to_numpy(np.float64)
REF["ref_reverse_rate"] = dv["asof_pitcher_reverse_rate"].to_numpy(np.float64)
REF["ref_middle_rate"] = dv["asof_pitcher_middle_rate"].to_numpy(np.float64)
REF["ref_strike_rate"] = dv["asof_pitcher_strike_rate"].to_numpy(np.float64)
REF["ref_pitcher_n"] = dv["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
REF["ref_batter_n"] = dv["asof_batter_n"].fillna(0).to_numpy(np.float64)
REF["ref_batter_success"] = dv["asof_batter_success_rate"].fillna(0.5).to_numpy(np.float64)
REF["ref_prev1_game"] = dv["asof_pitcher_prev1_game_success_rate"].fillna(0.5).to_numpy(np.float64)
REF["ref_fastball_rate"] = dv["asof_pitcher_fastball_rate"].fillna(0.5).to_numpy(np.float64)
REF["ref_count_state"] = (dv["balls_before"] * 4 + dv["strikes_before"]).to_numpy(np.float64)
REF["ref_inning"] = dv["inning"].to_numpy(np.float64)

# ---------------- 후보: 의도축 (phase77 기각) ----------------
S = dv["asof_pitcher_success_rate"].to_numpy(np.float64)
R = dv["asof_pitcher_reverse_rate"].to_numpy(np.float64)
M = dv["asof_pitcher_middle_rate"].to_numpy(np.float64)
B = dv["asof_pitcher_ball_rate"].to_numpy(np.float64)
K = dv["asof_pitcher_strike_rate"].to_numpy(np.float64)
fail = np.clip(1.0 - S, 1e-6, None)
CAND["rev_share"] = R / fail
CAND["mid_share"] = M / fail
CAND["safe_share"] = 1.0 - (R + M) / fail
CAND["inplay_rate"] = 1.0 - B - K
CAND["chase_intent"] = B - fail
CAND["zone_minus_success"] = K - S

# ---------------- 후보: 시대보정 (phase87/88 기각) ----------------
league = df.groupby("season")["control_success"].mean().to_dict()
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)


def cnt(col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)


S_ = cnt("asof_pitcher_success_rate")
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
tmp = dv[["pitcher_id"]].join((hist["resid"] / hist["n_tot"]).rename("sk"), on="pitcher_id") \
                        .join((hist["s_tot"] / hist["n_tot"]).rename("nv"), on="pitcher_id")
CAND["era_skill"] = tmp["sk"].fillna(0.0).to_numpy(np.float64)
CAND["era_correction"] = (tmp["sk"] + league[2024] - tmp["nv"]).fillna(0.0).to_numpy(np.float64)

# ---------------- 후보: 투수x타자 쌍 (오라클 1.6점 기각) ----------------
h = df[df.season <= 2023]
gl = h["control_success"].mean()
pr = h.groupby("pitcher_id")["control_success"].agg(ps="sum", pn="count")
br = h.groupby("batter_id")["control_success"].agg(bs="sum", bn="count")
pair = h.groupby(["pitcher_id", "batter_id"])["control_success"].agg(s="sum", n="count")
t2 = dv[["pitcher_id", "batter_id"]].join(pr, on="pitcher_id").join(br, on="batter_id") \
        .join(pair, on=["pitcher_id", "batter_id"]).fillna({"ps": 0, "pn": 0, "bs": 0, "bn": 0, "s": 0, "n": 0})
pri = np.clip(((t2["ps"] + 1000 * gl) / (t2["pn"] + 1000)) +
              ((t2["bs"] + 1000 * gl) / (t2["bn"] + 1000)) - gl, 0.02, 0.98).to_numpy(float)
CAND["pair_diff_K500"] = (t2["s"].to_numpy(float) + 500 * pri) / (t2["n"].to_numpy(float) + 500) - pri

# ---------------- 후보: 트릭맨 실행실패 (phase82, 최고 4.44점) ----------------
log("트릭맨 로드 (tagged vs auto 불일치)...")
USE = ["season", "pitch_of_pa", "balls_before", "pitcher_trackman_id",
       "tagged_pitch_type", "auto_pitch_type", "rel_speed", "zone_speed"]
m = pd.read_csv(MAP_PATH).sort_values("sim", ascending=False).drop_duplicates("tm_id")
t2p = m.set_index("tm_id")["pitcher_id"]
tm = pd.read_csv(TM_PATH, encoding="utf-8-sig", usecols=USE)
tm = tm.rename(columns={"pitcher_trackman_id": "tm_id"})
tm["pitcher_id"] = tm["tm_id"].map(t2p)
tm = tm.dropna(subset=["pitcher_id"])
tm["pitcher_id"] = tm["pitcher_id"].astype(np.int64)
tg = tm["tagged_pitch_type"].astype(str).str.strip().str.lower()
au = tm["auto_pitch_type"].astype(str).str.strip().str.lower()
valid = (~tg.isin(["nan", "undefined", "other", ""])) & (~au.isin(["nan", "undefined", "other", ""]))
tm["disagree"] = np.where(valid, (tg != au).astype(float), np.nan)
tm["velo_loss"] = tm["rel_speed"] - tm["zone_speed"]
tm["deep_pa"] = (tm["pitch_of_pa"] >= 4).astype(float)
tm["is3b"] = (tm["balls_before"] >= 3).astype(float)
hist_tm = tm[tm.season <= 2023]
agg = hist_tm.groupby("pitcher_id").agg(n=("rel_speed", "size"), dis=("disagree", "mean"),
                                        vls=("velo_loss", "std"), vlm=("velo_loss", "mean"))
d_deep = hist_tm[hist_tm.deep_pa == 1].groupby("pitcher_id")["disagree"].mean().rename("dis_deep")
d_pr = hist_tm[hist_tm.is3b == 1].groupby("pitcher_id")["disagree"].mean().rename("dis_press")
agg = agg.join(d_deep).join(d_pr)
agg["disagree_deep"] = agg["dis_deep"] - agg["dis"]
agg["disagree_press"] = agg["dis_press"] - agg["dis"]
tt = dv[["pitcher_id"]].join(agg, on="pitcher_id")
for c, nm in [("disagree_deep", "tm_disagree_deep"), ("disagree_press", "tm_disagree_press"),
              ("dis", "tm_disagree_raw"), ("vls", "tm_velo_loss_sd"), ("vlm", "tm_velo_loss_mean")]:
    v = tt[c].to_numpy(np.float64)
    CAND[nm] = np.nan_to_num(v, nan=np.nanmedian(v))
log(f"  트릭맨 매칭율 {tt['n'].notna().mean():.3f}")

log(f"참조 {len(REF)}개, 후보 {len(CAND)}개 스크리닝...")
rows = []
for grp, dct in [("참조", REF), ("후보", CAND)]:
    for name, v in dct.items():
        lg_ = linear_gain(y_va, p_gbdt, v)
        sh, sd = splithalf_gain(ry, v)
        rows.append(dict(group=grp, feature=name, linear=lg_, splithalf=sh, sd=sd,
                         corr_p=abs(np.corrcoef(_clean(v), p_gbdt)[0, 1])))
res = pd.DataFrame(rows)

print()
print("=" * 82)
print("참조 피처 (모델이 이미 쓰는 것 -> 0 근처여야 스크리너 정상)")
print("-" * 82)
print(f"{'피처':<24}{'선형':>10}{'분할반':>10}{'SD':>8}{'|corr(z,p)|':>13}")
r_ref = res[res.group == "참조"].sort_values("splithalf", ascending=False)
for _, r in r_ref.iterrows():
    print(f"{r.feature:<24}{r.linear:10.2f}{r.splithalf:10.2f}{r.sd:8.2f}{r.corr_p:13.3f}")
ref_mean, ref_sd = r_ref.splithalf.mean(), r_ref.splithalf.std(ddof=1)
ref_max = r_ref.splithalf.max()
print(f"\n참조 분포: 평균 {ref_mean:+.2f}  SD {ref_sd:.2f}  최댓값 {ref_max:+.2f}")
thr = ref_mean + 2 * ref_sd
print(f"판정 문턱 (참조평균 + 2SD) = {thr:+.2f}")

print()
print("=" * 82)
print("후보 피처 (기각했던 것들)")
print("-" * 82)
print(f"{'피처':<24}{'선형':>10}{'분할반':>10}{'SD':>8}{'문턱초과':>10}{'|corr(z,p)|':>13}")
r_c = res[res.group == "후보"].sort_values("splithalf", ascending=False)
for _, r in r_c.iterrows():
    over = r.splithalf - thr
    mark = "  <-- 통과" if over > 0 else ""
    print(f"{r.feature:<24}{r.linear:10.2f}{r.splithalf:10.2f}{r.sd:8.2f}{over:+10.2f}{r.corr_p:13.3f}{mark}")

winners = r_c[r_c.splithalf > thr]
print()
if len(winners):
    print(f"문턱 통과: {list(winners.feature)}")
    print("-> 실제 모델에 넣어 SHAP/실측 검증 가치 있음")
else:
    print("문턱을 넘는 후보 없음 -> 기각한 피처들에 남은 정보 없음 (모델을 바꿔도 동일)")
res.to_csv("phase93_splithalf_screen.csv", index=False)
log(f"총 {time.time()-t0:.0f}s")
