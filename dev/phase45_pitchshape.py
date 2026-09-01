"""pitch-shape 군집 독립 검증 — (1) 자체 재계산으로 보고된 수치 재현 (2) v15 실제 잔차로 재확인.

보고된 수치: 진짜SD=0.01233, 재현상관 r=0.356 [0.310,0.407], 상한 +60.9
검증된 규칙: 잔차가치(v15 트레인모델 기준) <2 기각 / >5 채택후보 (구종 +5.6->실제+6.7로 검증됨)
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
from pitchshape import assign_clusters, build_matched_with_phys, build_shape_tables, fit_shape_clusters
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]

print("매칭 + 물리량...", flush=True)
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
matched = build_matched_with_phys(df)
print(f"  매칭 {len(matched):,}행 (커버리지 {len(matched)/len(df)*100:.2f}%)  ({time.time()-t0:.0f}s)", flush=True)

print("군집화...", flush=True)
models = fit_shape_clusters(matched)
shape_lab = assign_clusters(matched, models)
print(f"  군집 유효 {shape_lab.notna().sum():,}행  ({time.time()-t0:.0f}s)", flush=True)

print("자체 진짜SD 재계산...", flush=True)
d = matched[["pitcher_id", "ptype", "season", "control_success"]].copy()
d["shape"] = shape_lab
d = d.dropna(subset=["shape"])
cell = d.groupby(["pitcher_id", "shape"]).agg(n=("control_success", "count"), m=("control_success", "mean"))
cell = cell[cell.n >= 30]
obs_var = cell.m.var()
noise = (cell.m * (1 - cell.m) / cell.n).mean()
print(f"  cells={len(cell):,}  관측SD={np.sqrt(obs_var):.4f}  노이즈SD={np.sqrt(noise):.4f}  "
      f"진짜SD(대략)={np.sqrt(max(obs_var-noise,0)):.4f}  (보고값 0.01233과 비교)", flush=True)

g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())
se = build_season_end_table(df)
dins = transform_inseason(df, se, g, sr)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
dpt_old = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
gr = build_global_rates(df)
ly_tbl = build_lastyear_table(df)
dly = transform_lastyear(df, ly_tbl, gr, sr, k=30.0)

print("\nshape 피처 조회 테이블 구축...", flush=True)
tables = build_shape_tables(matched, shape_lab)
top_shape = (matched.assign(shape=shape_lab).dropna(subset=["shape"])
            .groupby("pitcher_id")["shape"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) else np.nan))


def transform_shape(df, tables, prior_rate, seasons_range):
    ctrl = tables["ctrl"]
    pid = df["pitcher_id"].to_numpy()
    season = df["season"].to_numpy()
    prev = season - 1
    n_rows = len(df)
    prior = np.asarray(prior_rate, np.float64)
    df_shape = df["pitcher_id"].map(top_shape)
    piv_s = (ctrl.pivot_table(index=["pitcher_id", "shape"], columns="season", values="s", aggfunc="first")
             .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True))
    piv_n = (ctrl.pivot_table(index=["pitcher_id", "shape"], columns="season", values="n", aggfunc="first")
             .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True))
    idx_ = pd.MultiIndex.from_arrays([pid, df_shape.to_numpy(), prev])
    s_ = np.nan_to_num(piv_s.reindex(idx_).to_numpy().astype(np.float64), nan=0.0)
    n_ = np.nan_to_num(piv_n.reindex(idx_).to_numpy().astype(np.float64), nan=0.0)
    rate = np.divide(s_, n_, out=np.full(n_rows, np.nan), where=n_ > 0)
    out = pd.DataFrame(index=df.index)
    out["shape_dev"] = np.nan_to_num(rate, nan=0.0) - prior
    out["shape_n"] = np.log1p(n_)
    return out


dshape = transform_shape(df, tables, pp, sr)
print(f"  shape_dev SD={dshape['shape_dev'].std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
ytr, yva = fold["y_train"], fold["y_valid"]
tr, va = df[df.season <= 2023].index, df[df.season == 2024].index
ti, ei = time_split_es(len(tr))


def stack(i, bf):
    X = pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt_old.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


Xtr, Xva = stack(tr, fold["X_train"]), stack(va, fold["X_valid"])
h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                   l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                   n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                        verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
cb.fit(Xtr.iloc[ti], ytr[ti], eval_set=(Xtr.iloc[ei], ytr[ei]))
p = 0.5 * h.predict_proba(Xva)[:, 1] + 0.5 * cb.predict_proba(Xva)[:, 1]
print(f"\nv15 구성 score={max(0, evaluate(yva, p)['bss']*1e5):.1f}", flush=True)

e = yva - p
r = yva.mean()
bv = r * (1 - r)
Xn = Xva.to_numpy(np.float64)
Xd = np.column_stack([np.ones(len(Xn)), Xn, p, p ** 2])
XtX = Xd.T @ Xd + 1e-3 * np.eye(Xd.shape[1])
zdf = dshape.loc[va]
keep = [c for c in zdf.columns if zdf[c].std() > 1e-12]
Z = zdf[keep].to_numpy(np.float64)
Zp = Z - Xd @ np.linalg.solve(XtX, Xd.T @ Z)
proj = Zp @ np.linalg.solve(Zp.T @ Zp + 1e-8 * np.eye(Zp.shape[1]), Zp.T @ e)
joint = (proj @ proj) / len(e) / bv * 1e5
mark = "채택후보" if joint >= 5 else ("애매" if joint >= 2 else "기각")
print(f"\n[pitch-shape] v15잔차 대비 합동가치 = {joint:+.1f}  ({mark})", flush=True)
for c in keep:
    z = zdf[c].to_numpy(np.float64)
    zp = z - Xd @ np.linalg.solve(XtX, Xd.T @ z)
    v = zp.var()
    if v < 1e-14:
        continue
    print(f"  {c:12s} corr={np.corrcoef(e, zp)[0,1]:+.5f}  개별={(np.cov(e,zp)[0,1]**2/v)/bv*1e5:+6.2f}", flush=True)
print(f"\n총 {time.time()-t0:.0f}s", flush=True)
