"""phase87 — 시대보정(era-adjusted) 커리어 레이트 스크리닝, 2폴드 검증.

배경: 리그 전체 성공률이 시즌마다 단조 하락한다 (2019 0.5647 -> 2024 0.4861, -0.0786).
그런데 asof_pitcher_success_rate는 투수의 커리어 누적이라, 언제 뛰었느냐에 따라
'섞인 시대'가 다르다. 2019년부터 뛴 베테랑은 고환경 시즌이 섞여 커리어 레이트가
위로 끌려 올라가 있고, 최근 데뷔한 신인은 저환경 시즌만 봐서 그대로 낮게 나온다.
즉 커리어 레이트가 진짜 실력 차이가 아니라 '데뷔 시점'을 일부 반영하고 있다.

시대보정 정의:
    skill_vs_era = sum_s (success_s - league_rate_s * n_s) / sum_s n_s
    era_adjusted = skill_vs_era + league_rate_target_season
    (naive career rate 대신, '이 투수가 각 시즌 리그 평균 대비 얼마나 잘했는지'를
     누적한 뒤, 예측 대상 시즌의 리그 평균에 다시 얹는다)

success/reverse/middle 세 축 모두에 적용, 그리고 naive와의 차이(=보정량 자체)도 후보에 넣는다.

방법: partial_gain (자유도1, 귀무편향 0.4점) + block_gain. today's rule: 반드시 2폴드
(fold A: train<=2023->2024, fold B: train<=2022->2023) 모두에서 확인되어야 채택.
fold A는 캐시된 HGB+CatBoost 블렌드 예측을 재사용, fold B는 신규 HGB 하나로 기준선을 만든다.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

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
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rz = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


def block_gain(y, p, Z):
    Z = np.column_stack([np.nan_to_num(Z[:, j], nan=np.nanmedian(Z[:, j])) for j in range(Z.shape[1])])
    Z = Z[:, [j for j in range(Z.shape[1]) if Z[:, j].std() > 0]]
    n, k = len(y), Z.shape[1]
    X0 = np.column_stack([np.ones(n), p])
    X1 = np.column_stack([X0, Z])

    def r2(X):
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - (y - X @ c).var() / y.var()

    return 1e5 * (r2(X1) - r2(X0) - k / n), k


log("데이터 로드...")
COLS = ["row_id", "season", "pitcher_id", "control_success", "asof_pitcher_n",
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate"]
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig", usecols=COLS)
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
sr = sorted(df["season"].unique().tolist())
y_all = df["control_success"].to_numpy(np.float64)

# ----------------------------------------------------------------------
# 리그 시즌별 평균 (train 전체에서, 각 시즌의 실제 관측치 — 정적 집계, 누수 아님)
# ----------------------------------------------------------------------
league_rate = df.groupby("season")["control_success"].mean().to_dict()
log(f"시즌별 리그 성공률: {[(s, round(league_rate[s], 4)) for s in sr]}")

# ----------------------------------------------------------------------
# 투수별 행 단위 라벨 복원 -> per-season 누적 (success/reverse/middle)
# ----------------------------------------------------------------------
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)


def cnt(col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)


S_, R_, M_ = [cnt(c) for c in ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                                "asof_pitcher_middle_rate"]]
ordr = df.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = df["pitcher_id"].to_numpy()[ordr]
sea_o = df["season"].to_numpy()[ordr]
n_o = n_[ordr]
step = np.zeros(len(df), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
s_diff = np.zeros(len(df)); r_diff = np.zeros(len(df)); m_diff = np.zeros(len(df))
s_diff[ordr[:-1]] = np.diff(S_[ordr])
r_diff[ordr[:-1]] = np.diff(R_[ordr])
m_diff[ordr[:-1]] = np.diff(M_[ordr])
log(f"행단위 라벨 복원 {step.sum():,}행 ({100*step.mean():.2f}%)")

lab = pd.DataFrame({"pid": df["pitcher_id"].to_numpy()[step], "season": df["season"].to_numpy()[step],
                    "s": s_diff[step], "r": r_diff[step], "m": m_diff[step], "n": np.ones(step.sum())})
per_season = lab.groupby(["pid", "season"]).agg(s=("s", "sum"), r=("r", "sum"),
                                                 m=("m", "sum"), n=("n", "sum")).reset_index()
for col in ["s", "r", "m"]:
    per_season[f"lg_{col}"] = per_season["season"].map(league_rate)
    per_season[f"resid_{col}"] = per_season[col] - per_season[f"lg_{col}"] * per_season["n"]
log(f"투수x시즌 셀 {len(per_season):,}개")


def build_era_features(train_upto, target_season):
    """train_upto까지의 이력으로 target_season 행에 붙일 시대보정 피처를 만든다."""
    hist = per_season[per_season.season <= train_upto]
    agg = hist.groupby("pid").agg(n_tot=("n", "sum"), resid_s=("resid_s", "sum"),
                                  resid_r=("resid_r", "sum"), resid_m=("resid_m", "sum"),
                                  s_tot=("s", "sum"), r_tot=("r", "sum"), m_tot=("m", "sum"))
    lg_t = league_rate.get(target_season, np.mean(list(league_rate.values())))
    out = pd.DataFrame(index=agg.index)
    for col in ["s", "r", "m"]:
        skill = agg[f"resid_{col}"] / agg["n_tot"].replace(0, np.nan)
        out[f"era_adj_{col}"] = (skill + lg_t).fillna(lg_t)
        out[f"naive_{col}"] = (agg[f"{col}_tot"] / agg["n_tot"].replace(0, np.nan)).fillna(lg_t)
        out[f"era_correction_{col}"] = out[f"era_adj_{col}"] - out[f"naive_{col}"]
    out["era_n"] = agg["n_tot"]
    return out


CAND_COLS = [f"{p}_{c}" for p in ["era_adj", "naive", "era_correction"] for c in ["s", "r", "m"]]


def run_fold(train_upto, valid_season, p_base, y_va, tag, mask_va=None):
    log(f"=== fold {tag}: train<={train_upto} -> valid={valid_season} ===")
    feat_tbl = build_era_features(train_upto, valid_season)
    va = df[df.season == valid_season][["pitcher_id"]].copy()
    joined = va.join(feat_tbl, on="pitcher_id")
    lg_t = league_rate.get(valid_season, np.mean(list(league_rate.values())))
    for c in CAND_COLS:
        joined[c] = joined[c].fillna(lg_t if "correction" not in c else 0.0)
    if mask_va is not None:
        joined = joined[mask_va].reset_index(drop=True)

    rows = []
    for c in CAND_COLS:
        gn, pc = partial_gain(y_va, p_base, joined[c].to_numpy(np.float64))
        rows.append(dict(feature=c, gain=gn, sigma=abs(pc) * np.sqrt(len(y_va)), sign=np.sign(pc)))
    res = pd.DataFrame(rows).sort_values("gain", ascending=False)

    Z_corr = np.column_stack([joined[f"era_correction_{c}"].to_numpy(np.float64) for c in ["s", "r", "m"]])
    g_corr, k_corr = block_gain(y_va, p_base, Z_corr)
    Z_adj = np.column_stack([joined[f"era_adj_{c}"].to_numpy(np.float64) for c in ["s", "r", "m"]])
    g_adj, k_adj = block_gain(y_va, p_base, Z_adj)

    print(f"\n[fold {tag}]")
    print(res.to_string(index=False))
    print(f"블록(era_correction x3): {g_corr:+.2f} (k={k_corr})")
    print(f"블록(era_adj x3):        {g_adj:+.2f} (k={k_adj})")
    return res, g_corr, g_adj


# fold A: 캐시 재사용
log("fold A 기준선 로드 (캐시)...")
zc = np.load("phase67_cache/phase69_preds.npz")
p_gbdt_A = 0.5 * zc["hgb"] + 0.5 * zc["cat3"]
y_A = zc["y"].astype(np.float64)
resA, gcA, gaA = run_fold(2023, 2024, p_gbdt_A, y_A, "A(2023->2024)")

# fold B: 신규 기준선 (HGB 하나, 빠르게)
log("fold B 기준선 학습 (HGB, train<=2022 -> 2023)...")
from features import FeatureBuilder, TARGET_COL
full = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
full["row_num"] = full["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
fb2 = FeatureBuilder(seed=42, include_raw_rates=False, team_te_mode="expanding").fit(full)
Xb = fb2.transform_train_oof(full).reset_index(drop=True)
tr_m = (full["season"] <= 2022).to_numpy()
va_m = (full["season"] == 2023).to_numpy()
yv_B = full[TARGET_COL].to_numpy(np.float64)[va_m]
w = 0.5 ** ((2022 - full["season"].to_numpy(np.float64)[tr_m]) / 2.0)
hgb = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                     l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                     n_iter_no_change=20, random_state=42)
hgb.fit(Xb.loc[tr_m], full[TARGET_COL].to_numpy()[tr_m], sample_weight=w)
p_base_B = hgb.predict_proba(Xb.loc[va_m])[:, 1]
log(f"  fold B 기준선 score={1e5*(1-np.mean((p_base_B-yv_B)**2)/(yv_B.mean()*(1-yv_B.mean()))):.2f}")

resB, gcB, gaB = run_fold(2022, 2023, p_base_B, yv_B, "B(2022->2023)")

print()
print("=" * 60)
print(f"era_correction 블록: fold A {gcA:+.2f}  fold B {gcB:+.2f}")
print(f"era_adj 블록:        fold A {gaA:+.2f}  fold B {gaB:+.2f}")
both_ok = (gcA > 3 and gcB > 3) or (gaA > 3 and gaB > 3)
print("=> 두 폴드 모두 신호 있음: 채택 검토" if both_ok else "=> 한쪽 이상 신호 없음: 기각")
log(f"총 {time.time()-t0:.0f}s")
