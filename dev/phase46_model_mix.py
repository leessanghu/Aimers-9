"""v15 피처셋에서 모델 다양성 재검증 — XGB/LGBM 추가 + CatBoost 시드평균.

배경 (전부 실측):
  v14a(50:50)=970.493 vs v14b(20:80)=970.546 -> 가중치 이동 효과 +0.05 (거의 없음)
  교차항 이전 시절: LGBM 772 / XGB 759 / Cat 806, hgb+lgbm+cat(827) < hgb+cat(841)
  -> 하지만 피처셋이 크게 바뀌었으므로 재확인 가치 있음

측정:
  1) 4모델(HGB/Cat/LGBM/XGB) 개별 성능
  2) 예측 상관 (낮을수록 앙상블 이득 실재)
  3) CatBoost 3시드 평균 (편향 불변, 분산만 감소 - 이론적으로 손해 불가)
  4) 조합별 점수 + 최적 가중치 탐색
"""
import sys
import time
from itertools import combinations

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
print(f"v15 피처 준비 ({time.time()-t0:.0f}s)", flush=True)

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
print(f"{Xtr.shape[1]}피처  train={len(Xtr):,}  valid={len(Xva):,}", flush=True)
sc = lambda p: max(0, evaluate(yva, p)["bss"] * 1e5)
preds = {}

t = time.time()
h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                   l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
preds["hgb"] = h.predict_proba(Xva)[:, 1]
print(f"  hgb   {sc(preds['hgb']):7.1f}  ({time.time()-t:.0f}s)", flush=True)

for s in (42, 7, 2024):
    t = time.time()
    c = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                           random_seed=s, verbose=0, early_stopping_rounds=50,
                           min_data_in_leaf=200, loss_function="Logloss")
    c.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
    preds[f"cat{s}"] = c.predict_proba(Xva)[:, 1]
    print(f"  cat{s:<5d} {sc(preds[f'cat{s}']):7.1f}  ({time.time()-t:.0f}s)", flush=True)

from lightgbm import LGBMClassifier, early_stopping, log_evaluation
t = time.time()
lg = LGBMClassifier(n_estimators=3000, learning_rate=0.03, num_leaves=31, max_depth=6,
                    min_child_samples=200, reg_lambda=5.0, colsample_bytree=0.8, subsample=0.9,
                    subsample_freq=1, random_state=SEED, n_jobs=-1, verbosity=-1)
lg.fit(Xtr.iloc[ti], ytr[ti], eval_set=[(Xtr.iloc[ei], ytr[ei])],
       callbacks=[early_stopping(50, verbose=False), log_evaluation(0)])
preds["lgbm"] = lg.predict_proba(Xva)[:, 1]
print(f"  lgbm  {sc(preds['lgbm']):7.1f}  iter={lg.best_iteration_}  ({time.time()-t:.0f}s)", flush=True)

from xgboost import XGBClassifier
t = time.time()
xg = XGBClassifier(n_estimators=3000, learning_rate=0.03, max_depth=6, min_child_weight=200,
                   reg_lambda=5.0, subsample=0.9, colsample_bytree=0.8, random_state=SEED,
                   n_jobs=-1, eval_metric="logloss", early_stopping_rounds=50, tree_method="hist")
xg.fit(Xtr.iloc[ti], ytr[ti], eval_set=[(Xtr.iloc[ei], ytr[ei])], verbose=False)
preds["xgb"] = xg.predict_proba(Xva)[:, 1]
print(f"  xgb   {sc(preds['xgb']):7.1f}  iter={xg.best_iteration}  ({time.time()-t:.0f}s)", flush=True)

cat_avg = np.mean([preds[f"cat{s}"] for s in (42, 7, 2024)], axis=0)
print(f"\n  cat 3시드평균 {sc(cat_avg):7.1f}  (단일 cat42={sc(preds['cat42']):.1f})", flush=True)

print("\n" + "=" * 70 + "\n예측 상관 (낮을수록 앙상블 이득 실재)\n" + "=" * 70, flush=True)
main = {"hgb": preds["hgb"], "cat": cat_avg, "lgbm": preds["lgbm"], "xgb": preds["xgb"]}
for a, b in combinations(main, 2):
    print(f"  {a:5s}-{b:5s}  r={np.corrcoef(main[a], main[b])[0,1]:.4f}", flush=True)

print("\n" + "=" * 70 + "\n조합별 점수 (v15 실제구성 = hgb0.5+cat단일0.5)\n" + "=" * 70, flush=True)
base_v15 = sc(0.5 * preds["hgb"] + 0.5 * preds["cat42"])
print(f"  [기준] v15 실제구성        {base_v15:7.1f}", flush=True)
rows = [
    ("hgb.5 + cat3시드.5", 0.5 * preds["hgb"] + 0.5 * cat_avg),
    ("hgb.3 + cat3시드.7", 0.3 * preds["hgb"] + 0.7 * cat_avg),
    ("hgb.2 + cat3시드.8", 0.2 * preds["hgb"] + 0.8 * cat_avg),
    ("cat3시드 단독", cat_avg),
    ("hgb.25+cat.5+lgbm.25", 0.25 * preds["hgb"] + 0.5 * cat_avg + 0.25 * preds["lgbm"]),
    ("hgb.25+cat.5+xgb.25", 0.25 * preds["hgb"] + 0.5 * cat_avg + 0.25 * preds["xgb"]),
    ("hgb.2+cat.5+lgbm.15+xgb.15", 0.2 * preds["hgb"] + 0.5 * cat_avg + 0.15 * preds["lgbm"] + 0.15 * preds["xgb"]),
    ("4모델 균등", 0.25 * (preds["hgb"] + cat_avg + preds["lgbm"] + preds["xgb"])),
]
for nm, p in sorted(rows, key=lambda x: -sc(x[1])):
    print(f"  {nm:28s} {sc(p):7.1f}   기준대비 {sc(p)-base_v15:+6.1f}", flush=True)
print(f"\n총 {time.time()-t0:.0f}s", flush=True)
