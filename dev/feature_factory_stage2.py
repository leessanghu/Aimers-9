"""2단계 — phase93 정확히 동일한 split-half 비편향 스크리너로 OOF141/OOF151 재검증.
1단계에서 gain +92.98(OOF151), +28.11(OOF141)로 크게 나왔지만 평균지지표본이
작아(353, 687) 오늘 밤 계속 봤던 불안정 패턴과 같은 프로필. phase93의 참조피처
12개와 나란히 놓고 판단한다.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

VALID_SEASON = 2024
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
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
df = df.sort_values("row_num").reset_index(drop=True)
y_all = df["control_success"].to_numpy(np.float64)
g = float(y_all.mean())

zc = np.load("phase67_cache/phase69_preds.npz")
p_gbdt = 0.5 * zc["hgb"] + 0.5 * zc["cat3"]
y_va = zc["y"].astype(np.float64)
va = (df["season"] == VALID_SEASON).to_numpy()
assert va.sum() == len(y_va)
ry = residualize_np(y_va, p_gbdt, n_bins=200)
log(f"valid={va.sum():,}  비모수 잔차 준비")

dv = df[va].reset_index(drop=True)
REF = {}
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

log("1단계 생존자 전체 재구성 (OOF는 K=20/50, TF는 shifted expanding)...")
CAND = {}

OOF_SURVIVORS = [
    ("OOF141", ["pitcher_id", "season"]),
    ("OOF151", ["pitcher_id", "season", "batter_hand"]),
    ("OOF161", ["batter_id", "season", "pitcher_hand"]),
    ("OOF041", ["pitcher_hand", "batter_hand", "base_state", "outs_before"]),
    ("OOF031", ["pitcher_hand", "batter_hand", "balls_before", "strikes_before"]),
    ("OOF051", ["pitcher_team_id", "game_type", "balls_before", "strikes_before"]),
    ("OOF061", ["batter_team_id", "game_type", "balls_before", "strikes_before"]),
    ("OOF101", ["season", "game_month", "pitcher_hand", "batter_hand"]),
    ("OOF001", ["season", "game_type", "balls_before", "strikes_before"]),
]
for fam_id, keys in OOF_SURVIVORS:
    grp = df.groupby(keys)["control_success"]
    cs = grp.cumsum() - df["control_success"]
    cn = grp.cumcount()
    for K in [20.0, 50.0]:
        rate = (cs + K * g) / (cn + K)
        CAND[f"{fam_id}_K{int(K)}"] = rate.to_numpy(np.float64)[va]

# TF081_std (1단계 유일한 TF 생존자, gain=5.28)
grp_tf = df.groupby(["season", "game_month", "pitcher_hand"])["li"]
tf081_std = grp_tf.apply(lambda s: s.shift(1).expanding().std())
CAND["TF081_std"] = np.asarray(tf081_std)[va] if not hasattr(tf081_std, "to_numpy") \
    else tf081_std.to_numpy(np.float64)[va]

log(f"참조 {len(REF)}개, 후보 {len(CAND)}개 split-half 스크리닝...")
rows = []
for grp_name, dct in [("참조", REF), ("후보", CAND)]:
    for name, v in dct.items():
        lg_ = linear_gain(y_va, p_gbdt, v)
        sh, sd = splithalf_gain(ry, v)
        rows.append(dict(group=grp_name, feature=name, linear=lg_, splithalf=sh, sd=sd))
        log(f"  [{grp_name}] {name}: linear={lg_:.2f}  splithalf={sh:+.2f}±{sd:.2f}")

res = pd.DataFrame(rows)
ref_mean = res[res.group == "참조"]["splithalf"].mean()
ref_sd = res[res.group == "참조"]["splithalf"].std()
threshold = ref_mean + 2 * ref_sd
print()
print("=" * 90)
print(res.to_string(index=False))
print(f"\n참조피처 splithalf 평균={ref_mean:.2f}  SD={ref_sd:.2f}  판정기준(평균+2SD)={threshold:.2f}")
for _, r in res[res.group == "후보"].iterrows():
    verdict = "통과(채택후보)" if r["splithalf"] > threshold else "기각"
    print(f"  {r['feature']}: splithalf={r['splithalf']:+.2f}  ->  {verdict}")
log(f"총 {time.time()-t0:.0f}s")
