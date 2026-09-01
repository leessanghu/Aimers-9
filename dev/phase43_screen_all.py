"""후보 6종 일괄 스크리닝 (학습 없이 잔차 지표). baseline = v15 구성(976.099 실증).

검증된 지표(phase39): 구종 +5.6 -> 실제 +6.7 / workload,form +0.4 -> 실제 가치없음
판정: 합동 <2 기각 / >5 검증가치

(A) 투구단위 라벨 복원 x 볼카운트   <- 신규 발견. reverse/middle/ball/strike 라벨을 되살려
                                     '성공률'이 아닌 '실패유형별' 조건부를 만든다
(B) 투구단위 라벨 복원 x 이닝
(C) lastyear strike 축              <- 지금 success/reverse/ball/middle만 쓰고 strike는 미사용
(D) lastyear pitchmix 차분          <- pitchmix에 in-season 트릭의 '작년' 버전 적용
(E) arsenal stability (JS divergence) <- 구종 '구성 자체'의 변화. 물리량 실패와는 다른 종류
(F) Trackman pitch_of_pa 기반 command <- train에 없는 축
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchlabels import (LABELS, build_cond_table, build_global_offsets, recover_pitch_labels,
                         transform_cond_labels)
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())
se = build_season_end_table(df)
dins = transform_inseason(df, se, g, sr)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
gr = build_global_rates(df)
ly_tbl = build_lastyear_table(df)
dly = transform_lastyear(df, ly_tbl, gr, sr, k=30.0)
print(f"v15 피처 준비 ({time.time()-t0:.0f}s)", flush=True)

# ---------- (A)(B) 투구단위 라벨 복원 기반 조건부 ----------
labels = recover_pitch_labels(df)
print(f"라벨 복원: 유효 {labels['lab_reverse'].notna().mean()*100:.2f}%  "
      f"reverse평균={labels['lab_reverse'].mean():.4f}  ({time.time()-t0:.0f}s)", flush=True)
cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
inn = np.clip(df["inning"].to_numpy(np.int64), 1, 9)

tbl_c = build_cond_table(df, labels, cs)
gl_c, bc_c = build_global_offsets(df, labels, cs)
dlab_c = transform_cond_labels(df, tbl_c, gl_c, bc_c, cs, "lc", sr, k=400.0)

tbl_i = build_cond_table(df, labels, inn)
gl_i, bc_i = build_global_offsets(df, labels, inn)
dlab_i = transform_cond_labels(df, tbl_i, gl_i, bc_i, inn, "li", sr, k=400.0)
print(f"라벨 조건부 완료  lc_reverse_dev SD={dlab_c['lc_reverse_dev'].std():.5f}  "
      f"({time.time()-t0:.0f}s)", flush=True)

# ---------- (C) lastyear strike ----------
sub = df.sort_values(["pitcher_id", "row_num"])
last = sub.groupby(["pitcher_id", "season"], as_index=False).last()
nb = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
K_end = np.round(last["asof_pitcher_strike_rate"].fillna(0).to_numpy(np.float64) * nb)
kt = pd.DataFrame({"pitcher_id": last["pitcher_id"], "season": last["season"],
                   "K_end": K_end, "N_end": nb + 1})
pk = {c: kt.pivot(index="pitcher_id", columns="season", values=c)
        .reindex(columns=sr).ffill(axis=1).stack(future_stack=True) for c in ["K_end", "N_end"]}
i1 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
i2 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 2])
gk = float(np.average(df["asof_pitcher_strike_rate"].fillna(0), weights=df["asof_pitcher_n"].fillna(0) + 1))
k1 = {c: np.nan_to_num(pk[c].reindex(i1).to_numpy().astype(np.float64), nan=0.0) for c in pk}
k2 = {c: np.nan_to_num(pk[c].reindex(i2).to_numpy().astype(np.float64), nan=0.0) for c in pk}
n_ly = np.clip(k1["N_end"] - k2["N_end"], 0, None)
c_ly = np.clip(k1["K_end"] - k2["K_end"], 0, None)
raw = np.divide(c_ly, n_ly, out=np.full_like(n_ly, np.nan), where=n_ly > 0)
dstrike = pd.DataFrame({"ly_strike": (n_ly * np.nan_to_num(raw, nan=gk) + 30.0 * gk) / (n_ly + 30.0)},
                       index=df.index)

# ---------- (D) lastyear pitchmix + (E) arsenal stability ----------
MIX = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
nmb = last["asof_pitcher_pitchmix_n"].fillna(0).to_numpy(np.float64)
mt = {"pitcher_id": last["pitcher_id"], "season": last["season"], "MN": nmb}
for c in MIX:
    mt[c] = np.round(last[c].fillna(0).to_numpy(np.float64) * nmb)
mtd = pd.DataFrame(mt)
pm = {c: mtd.pivot(index="pitcher_id", columns="season", values=c)
        .reindex(columns=sr).ffill(axis=1).stack(future_stack=True) for c in ["MN"] + MIX}
m1 = {c: np.nan_to_num(pm[c].reindex(i1).to_numpy().astype(np.float64), nan=0.0) for c in pm}
m2 = {c: np.nan_to_num(pm[c].reindex(i2).to_numpy().astype(np.float64), nan=0.0) for c in pm}
mn_ly = np.clip(m1["MN"] - m2["MN"], 0, None)
gmix = {c: float(np.average(df[c].fillna(0), weights=df["asof_pitcher_pitchmix_n"].fillna(0) + 1)) for c in MIX}
dmix = pd.DataFrame(index=df.index)
ly_p, car_p = [], []
for c in MIX:
    cc = np.clip(m1[c] - m2[c], 0, None)
    r_ly = np.divide(cc, mn_ly, out=np.full_like(mn_ly, np.nan), where=mn_ly > 0)
    r_ly = (mn_ly * np.nan_to_num(r_ly, nan=gmix[c]) + 30.0 * gmix[c]) / (mn_ly + 30.0)
    r_car = np.divide(m1[c], m1["MN"], out=np.full_like(mn_ly, np.nan), where=m1["MN"] > 0)
    r_car = np.nan_to_num(r_car, nan=gmix[c])
    short = c.split("_")[-2]
    dmix[f"lymix_{short}"] = r_ly
    dmix[f"lymix_{short}_minus_career"] = r_ly - r_car
    ly_p.append(r_ly)
    car_p.append(r_car)
P = np.clip(np.vstack(ly_p).T, 1e-9, None); P /= P.sum(1, keepdims=True)
Q = np.clip(np.vstack(car_p).T, 1e-9, None); Q /= Q.sum(1, keepdims=True)
M = 0.5 * (P + Q)
js = 0.5 * (P * np.log(P / M)).sum(1) + 0.5 * (Q * np.log(Q / M)).sum(1)
dmix["arsenal_js"] = np.nan_to_num(js, nan=0.0)

# ---------- (F) Trackman pitch_of_pa ----------
try:
    pm_map = pd.read_csv("pitcher_map.csv").sort_values("sim", ascending=False).drop_duplicates("tm_id")
    p2t = pm_map.set_index("pitcher_id")["tm_id"]
    tmh = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig",
                      usecols=["season", "pitcher_trackman_id", "pitch_of_pa", "balls_before", "strikes_before"])
    tmh = tmh.rename(columns={"pitcher_trackman_id": "tm_id"})
    inv = p2t.reset_index().set_index("tm_id")["pitcher_id"]
    tmh["pitcher_id"] = tmh["tm_id"].map(inv)
    tmh = tmh.dropna(subset=["pitcher_id"])
    tmh["pitcher_id"] = tmh["pitcher_id"].astype(int)
    tmh["_cs"] = tmh["balls_before"] * 4 + tmh["strikes_before"]
    prof = (tmh.groupby(["pitcher_id", "season", "_cs"])["pitch_of_pa"]
            .agg(popa_mean="mean", popa_max="max").reset_index())
    ppv = {c: prof.pivot_table(index=["pitcher_id", "_cs"], columns="season", values=c, aggfunc="first")
             .reindex(columns=sr).ffill(axis=1).stack(future_stack=True) for c in ["popa_mean", "popa_max"]}
    pidx = pd.MultiIndex.from_arrays([df["pitcher_id"], cs, df["season"] - 1])
    dpopa = pd.DataFrame({c: np.nan_to_num(ppv[c].reindex(pidx).to_numpy().astype(np.float64), nan=0.0)
                          for c in ppv}, index=df.index)
    print(f"pitch_of_pa 프로필 완료 ({time.time()-t0:.0f}s)", flush=True)
except Exception as ex:
    print(f"pitch_of_pa 실패: {ex}", flush=True)
    dpopa = pd.DataFrame(index=df.index)

# ---------- v15 잔차 ----------
fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
ytr, yva = fold["y_train"], fold["y_valid"]
tr, va = df[df.season <= 2023].index, df[df.season == 2024].index
ti, ei = time_split_es(len(tr))


def stack(i, bf):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


Xtr, Xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                   l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                        verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
p = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
print(f"\nv15 구성 score={max(0, evaluate(yva, p)['bss']*1e5):.1f}  ({time.time()-t0:.0f}s)", flush=True)

e = yva - p
r = yva.mean()
bv = r * (1 - r)
Xn = Xva.to_numpy(np.float64)
Xd = np.column_stack([np.ones(len(Xn)), Xn, p, p ** 2])
XtX = Xd.T @ Xd + 1e-3 * np.eye(Xd.shape[1])


def screen(zdf, tag, detail=True):
    zdf = zdf.loc[va]
    keep = [c for c in zdf.columns if zdf[c].std() > 1e-12]
    if not keep:
        print(f"  [{tag}] 유효 피처 없음", flush=True)
        return
    Z = zdf[keep].to_numpy(np.float64)
    Zp = Z - Xd @ np.linalg.solve(XtX, Xd.T @ Z)
    proj = Zp @ np.linalg.solve(Zp.T @ Zp + 1e-8 * np.eye(Zp.shape[1]), Zp.T @ e)
    joint = (proj @ proj) / len(e) / bv * 1e5
    mark = "  ***채택후보***" if joint >= 5 else ("  (애매)" if joint >= 2 else "  기각")
    print(f"\n  [{tag}]  합동 = {joint:+.1f}{mark}", flush=True)
    if detail:
        for c in keep:
            z = zdf[c].to_numpy(np.float64)
            zp = z - Xd @ np.linalg.solve(XtX, Xd.T @ z)
            v = zp.var()
            if v < 1e-14:
                continue
            print(f"    {c:26s} corr={np.corrcoef(e, zp)[0,1]:+.5f}  개별={(np.cov(e,zp)[0,1]**2/v)/bv*1e5:+6.2f}", flush=True)


print("\n" + "=" * 78 + "\n후보 스크리닝 (규칙: 합동 <2 기각 / >5 채택후보)\n" + "=" * 78, flush=True)
screen(dlab_c, "(A) 투구단위라벨 x 볼카운트")
screen(dlab_i, "(B) 투구단위라벨 x 이닝")
screen(dstrike, "(C) lastyear strike")
screen(dmix, "(D)(E) lastyear pitchmix + arsenal JS")
if len(dpopa.columns):
    screen(dpopa, "(F) Trackman pitch_of_pa")
print("\n" + "=" * 78, flush=True)
allz = pd.concat([dlab_c, dlab_i, dstrike, dmix, dpopa], axis=1)
screen(allz, "전체 합동", detail=False)
print(f"\n총 {time.time()-t0:.0f}s", flush=True)
