"""phase88 — F리그 regime 단절을 반영한 시대보정, 2폴드 스크리닝.

phase87에서 순수 시즌단위 시대보정(era_correction)은 약했다(fold A +3.18, fold B +0.92).
그런데 fold B 기준선이 -1600.45로 깨졌던 원인이 F리그 2022->2023 단절
(0.709->0.473)이었다. 즉 시즌 단위 리그평균은 R/F를 섞어버려서 이 단절을 못 잡는다.

이번엔 (시즌, game_type) 셀 단위로 리그 기준선을 잡는다:
    regime_skill = sum_{s,gt} (success - league_rate[s,gt]*n) / sum(n)   (누적, train_upto까지)
    regime_adj   = regime_skill + league_rate[row의 실제 season, row의 실제 game_type]
    regime_correction = regime_adj - naive(시즌/게임타입 구분 없는 원시 커리어레이트)

regime_adj는 row마다 그 행의 실제 game_type을 그대로 쓰므로(공식 컬럼, row-internal) 규칙상
안전하다. success/reverse/middle 세 축 모두 적용.

방법: partial_gain 2폴드(A: train<=2023->2024, B: train<=2022->2023) 스크리닝.
통과하면 실제 162+3피처 모델을 학습해서 SHAP magnitude까지 확인한다(phase75 방식).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

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
COLS = ["row_id", "season", "game_type", "pitcher_id", "control_success", "asof_pitcher_n",
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate"]
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig", usecols=COLS)
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
sr = sorted(df["season"].unique().tolist())

league_sg = df.groupby(["season", "game_type"])["control_success"].mean().to_dict()
log("시즌x게임타입 리그율:")
for k in sorted(league_sg): print(f"    {k}: {league_sg[k]:.4f}")

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
s_diff = np.zeros(len(df)); r_diff = np.zeros(len(df)); m_diff = np.zeros(len(df))
s_diff[ordr[:-1]] = np.diff(S_[ordr]); r_diff[ordr[:-1]] = np.diff(R_[ordr]); m_diff[ordr[:-1]] = np.diff(M_[ordr])
log(f"행단위 라벨 복원 {step.sum():,}행 ({100*step.mean():.2f}%)")

lab = pd.DataFrame({"pid": df["pitcher_id"].to_numpy()[step], "season": df["season"].to_numpy()[step],
                    "gt": df["game_type"].to_numpy()[step],
                    "s": s_diff[step], "r": r_diff[step], "m": m_diff[step], "n": 1.0})
per_cell = lab.groupby(["pid", "season", "gt"]).agg(s=("s", "sum"), r=("r", "sum"),
                                                     m=("m", "sum"), n=("n", "sum")).reset_index()
for col in ["s", "r", "m"]:
    per_cell[f"lg_{col}"] = per_cell.apply(lambda row: league_sg.get((row["season"], row["gt"]), np.nan), axis=1)
    per_cell[f"resid_{col}"] = per_cell[col] - per_cell[f"lg_{col}"] * per_cell["n"]
log(f"투수x시즌x게임타입 셀 {len(per_cell):,}개")


def build_regime_features(train_upto):
    hist = per_cell[per_cell.season <= train_upto]
    agg = hist.groupby("pid").agg(n_tot=("n", "sum"), resid_s=("resid_s", "sum"),
                                  resid_r=("resid_r", "sum"), resid_m=("resid_m", "sum"),
                                  s_tot=("s", "sum"), r_tot=("r", "sum"), m_tot=("m", "sum"))
    return agg


def extrapolate_target_rate(train_upto, valid_season):
    """target season의 리그율을 '실제값'이 아니라 train_upto까지의 추세로 외삽한다.
    (실전에서는 예측대상 시즌의 진짜 리그율을 알 수 없다 -- 반드시 과거만으로 추정)"""
    hist = df[df.season <= train_upto]
    out = {}
    for gt in ["R", "F"]:
        sub = hist[hist.game_type == gt].groupby("season")["control_success"].mean()
        if len(sub) >= 2:
            x = sub.index.to_numpy(float); yv = sub.to_numpy(float)
            b = np.polyfit(x, yv, 1)
            out[gt] = float(np.clip(np.polyval(b, valid_season), 0.05, 0.95))
        else:
            out[gt] = float(sub.mean()) if len(sub) else float(hist["control_success"].mean())
    return out


def attach(df_va, agg, global_lg, lg_target_map):
    out = df_va[["pitcher_id", "season", "game_type"]].copy()
    out = out.join(agg, on="pitcher_id")
    lg_target_row = out["game_type"].map(lg_target_map).fillna(global_lg)
    for col in ["s", "r", "m"]:
        skill = out[f"resid_{col}"] / out["n_tot"].replace(0, np.nan)
        naive = (out[f"{col}_tot"] / out["n_tot"].replace(0, np.nan))
        out[f"regime_adj_{col}"] = (skill + lg_target_row).fillna(lg_target_row)
        out[f"naive2_{col}"] = naive.fillna(lg_target_row)
        out[f"regime_correction_{col}"] = out[f"regime_adj_{col}"] - out[f"naive2_{col}"]
    return out


CAND = [f"{p}_{c}" for p in ["regime_adj", "regime_correction"] for c in ["s", "r", "m"]]


def run_fold(train_upto, valid_season, p_base, y_va, tag):
    log(f"=== fold {tag} ===")
    agg = build_regime_features(train_upto)
    global_lg = df[df.season <= train_upto]["control_success"].mean()
    lg_target_map = extrapolate_target_rate(train_upto, valid_season)
    log(f"  외삽된 목표시즌({valid_season}) 리그율: {lg_target_map}  "
       f"(실제값: R={league_sg.get((valid_season,'R')):.4f} F={league_sg.get((valid_season,'F')):.4f})")
    va = df[df.season == valid_season]
    joined = attach(va, agg, global_lg, lg_target_map).reset_index(drop=True)

    rows = []
    for c in CAND:
        gn, pc = partial_gain(y_va, p_base, joined[c].to_numpy(np.float64))
        rows.append(dict(feature=c, gain=gn, sigma=abs(pc) * np.sqrt(len(y_va)), sign=np.sign(pc)))
    res = pd.DataFrame(rows).sort_values("gain", ascending=False)
    print(f"\n[fold {tag}]")
    print(res.to_string(index=False))

    Zc = np.column_stack([joined[f"regime_correction_{c}"].to_numpy(np.float64) for c in ["s", "r", "m"]])
    gc, kc = block_gain(y_va, p_base, Zc)
    Za = np.column_stack([joined[f"regime_adj_{c}"].to_numpy(np.float64) for c in ["s", "r", "m"]])
    ga, ka = block_gain(y_va, p_base, Za)
    print(f"블록(regime_correction x3): {gc:+.2f} (k={kc})")
    print(f"블록(regime_adj x3):        {ga:+.2f} (k={ka})")
    return gc, ga


log("fold A 기준선 로드 (캐시)...")
zc = np.load("phase67_cache/phase69_preds.npz")
p_gbdt_A = 0.5 * zc["hgb"] + 0.5 * zc["cat3"]
y_A = zc["y"].astype(np.float64)
gcA, gaA = run_fold(2023, 2024, p_gbdt_A, y_A, "A(2023->2024)")

log("fold B 기준선 학습 (HGB, train<=2022 -> 2023, 전체 162피처 재사용 시도)...")
from features import FeatureBuilder, TARGET_COL
from sklearn.ensemble import HistGradientBoostingClassifier
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
r_B = yv_B.mean()
log(f"  fold B 기준선 score={1e5*(1-np.mean((p_base_B-yv_B)**2)/(r_B*(1-r_B))):.2f}")

gcB, gaB = run_fold(2022, 2023, p_base_B, yv_B, "B(2022->2023)")

print()
print("=" * 60)
print(f"regime_correction 블록: fold A {gcA:+.2f}  fold B {gcB:+.2f}")
print(f"regime_adj 블록:        fold A {gaA:+.2f}  fold B {gaB:+.2f}")
ok = gcA > 3 and gcB > 3
print("=> 두 폴드 모두 순수신호 있음: 채택 검토" if ok else "=> 기각 또는 보류")
log(f"총 {time.time()-t0:.0f}s")
