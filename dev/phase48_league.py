"""리그 분리 피처 잔차 검증 — baseline = v15 구성(976.099 실증).

주의: 다변량 사영은 v17에서 부호 반전(+5.6 예측 -> 실제 -10.1)을 냈으므로 쓰지 않는다.
개별 피처 잔차가치(구종 3개 때 +5.6->실제+6.7로 맞았던 방식)만 신뢰한다.
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
from leaguesep import LG_COLS, build_league_table, league_global_rates, transform_league
from metrics import evaluate
from phase2_common import build_fold, time_split_es
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
dly = transform_lastyear(df, build_lastyear_table(df), gr, sr, k=30.0)

print("리그 분리 피처...", flush=True)
lg_tbl = build_league_table(df)
lg_glob = league_global_rates(df)
dlg = transform_league(df, lg_tbl, lg_glob, pp, sr)
print(f"  리그별 전역 성공률: { {k: round(v,4) for k,v in lg_glob.items()} }", flush=True)
m19 = df["season"] == 2019
print(f"  2019행 lg_own_n 최대={dlg.loc[m19,'lg_own_n'].max():.2e} (0이어야 정상)", flush=True)
print(f"  lg_diff SD={dlg['lg_diff'].std():.5f}  |  R행 lg_diff 평균={dlg.loc[df.game_type=='R','lg_diff'].mean():+.5f}"
      f"  F행 lg_diff 평균={dlg.loc[df.game_type=='F','lg_diff'].mean():+.5f}", flush=True)
print(f"  ({time.time()-t0:.0f}s)", flush=True)

fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
ytr, yva = fold["y_train"], fold["y_valid"]
tr, va = df[df.season <= 2023].index, df[df.season == 2024].index
ti, ei = time_split_es(len(tr))


def stack(i, bf, extra=()):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    parts = [X, add_crosses(X), dly.loc[i].reset_index(drop=True)]
    parts += [e.loc[i].reset_index(drop=True) for e in extra]
    return pd.concat(parts, axis=1)


Xtr, Xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                   l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                        verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
p = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
base = max(0, evaluate(yva, p)["bss"] * 1e5)
print(f"\nv15 구성 score={base:.1f}  ({time.time()-t0:.0f}s)", flush=True)

e = yva - p
r = yva.mean()
bv = r * (1 - r)
Xn = Xva.to_numpy(np.float64)
Xd = np.column_stack([np.ones(len(Xn)), Xn, p, p ** 2])
XtX = Xd.T @ Xd + 1e-3 * np.eye(Xd.shape[1])

print("\n=== 리그 피처 개별 잔차가치 (합산 아닌 개별만 신뢰) ===", flush=True)
tot = 0.0
for c in LG_COLS:
    z = dlg.loc[va, c].to_numpy(np.float64)
    if z.std() < 1e-12:
        continue
    zp = z - Xd @ np.linalg.solve(XtX, Xd.T @ z)
    v = zp.var()
    if v < 1e-14:
        print(f"  {c:14s} (전부 설명됨)", flush=True)
        continue
    gain = (np.cov(e, zp)[0, 1] ** 2 / v) / bv * 1e5
    tot += gain
    print(f"  {c:14s} z_perp SD={np.sqrt(v):.5f}  corr(e,z_perp)={np.corrcoef(e, zp)[0,1]:+.5f}  개별={gain:+6.2f}", flush=True)
print(f"  개별 합산(참고용) = {tot:+.1f}", flush=True)

# ---- 잔차 지표만 믿지 않고, 실제 폴드 학습으로도 즉시 확인 (4피처 추가라 저비용) ----
print("\n=== 폴드 직접 검증: v15 + 리그4피처 ===", flush=True)
Xtr2, Xva2 = stack(tr, fold["X_train"], [dlg]), stack(va, fold["X_valid"], [dlg])
h2 = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                    l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                    n_iter_no_change=20, random_state=SEED).fit(Xtr2, ytr)
ps = []
for s in (42, 7, 2024):
    c2 = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=s,
                            verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
    c2.fit(Xtr2.iloc[ti], ytr[ti], eval_set=(Xtr2.iloc[ei], ytr[ei]))
    ps.append(c2.predict_proba(Xva2)[:, 1])
p2 = 0.5 * h2.predict_proba(Xva2)[:, 1] + 0.5 * np.mean(ps, axis=0)
s2 = max(0, evaluate(yva, p2)["bss"] * 1e5)
print(f"  baseline(v15 단일시드)={base:.1f}  ->  +리그4 (cat3시드)={s2:.1f}", flush=True)
# 공정비교: 리그 없는 3시드도
ps0 = [cb.predict_proba(Xva)[:, 1]]
for s in (7, 2024):
    c0 = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=s,
                            verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
    c0.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
    ps0.append(c0.predict_proba(Xva)[:, 1])
p0 = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * np.mean(ps0, axis=0)
s0 = max(0, evaluate(yva, p0)["bss"] * 1e5)
print(f"  [공정비교] 리그없음+cat3시드={s0:.1f}  vs  리그4+cat3시드={s2:.1f}  ->  리그 순효과={s2-s0:+.1f}", flush=True)
print(f"\n총 {time.time()-t0:.0f}s", flush=True)
