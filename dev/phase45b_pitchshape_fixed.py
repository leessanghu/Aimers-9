"""phase45 버그 수정판.
(1) 보고된 delta 공식(r_pck - r_pc - r_ck + r_c)을 정확히 구현해 진짜SD 재현
(2) shape 피처를 pitchtype.py와 동일한 '주변화' 방식(단일 대표 shape 아님)으로 재구성
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
from pitchshape import assign_clusters, build_matched_with_phys, fit_shape_clusters
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
t0 = time.time()
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]

print("매칭 + 물리량 + 군집...", flush=True)
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
matched = build_matched_with_phys(df)
models = fit_shape_clusters(matched)
shape_lab = assign_clusters(matched, models)
print(f"  매칭 {len(matched):,}  ({time.time()-t0:.0f}s)", flush=True)

d = matched[["pitcher_id", "ptype", "season", "control_success"]].copy()
d["shape"] = shape_lab
d = d.dropna(subset=["shape"])

print("\n=== ANOVA 이중차분(delta) 정확 재현 ===", flush=True)
g_all = float(d["control_success"].mean())
r_pc = d.groupby(["pitcher_id", "ptype"])["control_success"].transform("mean")
r_ck = d.groupby(["ptype", "shape"])["control_success"].transform("mean")
r_c = d.groupby("ptype")["control_success"].transform("mean")
delta = d["control_success"] - r_pc - r_ck + r_c

cell = pd.DataFrame({"pitcher_id": d["pitcher_id"], "shape": d["shape"], "delta": delta}).groupby(
    ["pitcher_id", "shape"]).agg(n=("delta", "count"), m=("delta", "mean"))
cell = cell[cell["n"] >= 30]
obs_var = cell["m"].var()
p_hat = np.clip(cell["m"] + g_all, 1e-6, 1 - 1e-6)  # 근사 성공률로 되돌려서 노이즈 추정
noise = (p_hat * (1 - p_hat) / cell["n"]).mean()
true_var = max(obs_var - noise, 0.0)
print(f"  cells={len(cell):,}  관측SD={np.sqrt(obs_var):.5f}  노이즈SD={np.sqrt(noise):.5f}  "
      f"진짜SD={np.sqrt(true_var):.5f}  (보고값 0.01233)", flush=True)
print(f"  상한 = {true_var/(g_all*(1-g_all))*1e5:.1f}점  (보고값 +60.9)", flush=True)

print("\n=== 재현상관 (직전시즌 delta -> 다음시즌 delta) ===", flush=True)
seasons = sorted(d["season"].unique())
c2 = d.assign(delta=delta).groupby(["pitcher_id", "shape", "season"])["delta"].agg(m="mean", n="count")
DP, DN = [], []
for i in range(1, len(seasons)):
    S, T = seasons[i - 1], seasons[i]
    p_ = c2.xs(S, level="season") if S in c2.index.get_level_values("season") else None
    n_ = c2.xs(T, level="season") if T in c2.index.get_level_values("season") else None
    if p_ is None or n_ is None:
        continue
    j = p_.join(n_, how="inner", lsuffix="_p", rsuffix="_n")
    j = j[(j["n_p"] >= 30) & (j["n_n"] >= 20)]
    if len(j):
        DP.append(j["m_p"].to_numpy())
        DN.append(j["m_n"].to_numpy())
if DP:
    dp, dn = np.concatenate(DP), np.concatenate(DN)
    r = np.corrcoef(dp, dn)[0, 1]
    rng = np.random.default_rng(0)
    bs = [np.corrcoef(dp[i], dn[i])[0, 1] for i in (rng.integers(0, len(dp), len(dp)) for _ in range(400))]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  n={len(dp):,}  r={r:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]  (보고값 0.356 [0.310,0.407])", flush=True)

# ---------- v15 실제 잔차 검증 (주변화 방식으로 shape 피처 재구성) ----------
print("\n=== v15 잔차 대비 (주변화 방식 재구성) ===", flush=True)
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

# 주변화: pred = sum_shape P(shape|pitcher,ptype,season-1) * ctrl(pitcher,shape,season-1)
ctrl_tbl = d.assign(delta=delta).groupby(["pitcher_id", "shape", "season"])["control_success"].agg(
    s="sum", n="count").reset_index()
ctrl_tbl[["s", "n"]] = ctrl_tbl.groupby(["pitcher_id", "shape"])[["s", "n"]].cumsum()
freq_tbl = d.groupby(["pitcher_id", "shape", "season"]).size().rename("n").reset_index()
freq_tbl["n"] = freq_tbl.groupby(["pitcher_id", "shape"])["n"].cumsum()
shapes_all = sorted(d["shape"].dropna().unique())

K_CTRL = 340.0
K_MIX = 80.0
global_shape_rate = d.groupby("shape")["control_success"].mean().to_dict()
global_mix_n = d.groupby(["shape", "season"]).size().reset_index(name="n")
global_mix_n["n"] = global_mix_n.groupby("shape")["n"].cumsum()

pid = df["pitcher_id"].to_numpy()
season = df["season"].to_numpy()
prev = season - 1
n_rows = len(df)
num_pred = np.zeros(n_rows)
den_mix = np.zeros(n_rows)
tot_n = np.zeros(n_rows)


def piv_lookup(table, index_cols, value, seasons_range):
    p = table.pivot_table(index=index_cols, columns="season", values=value, aggfunc="first")
    return p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)


for sh in shapes_all:
    cs_ = np.nan_to_num(piv_lookup(ctrl_tbl[ctrl_tbl["shape"] == sh], "pitcher_id", "s", sr)
                        .reindex(pd.MultiIndex.from_arrays([pid, prev])).to_numpy().astype(np.float64), nan=0.0)
    cn_ = np.nan_to_num(piv_lookup(ctrl_tbl[ctrl_tbl["shape"] == sh], "pitcher_id", "n", sr)
                        .reindex(pd.MultiIndex.from_arrays([pid, prev])).to_numpy().astype(np.float64), nan=0.0)
    fn_ = np.nan_to_num(piv_lookup(freq_tbl[freq_tbl["shape"] == sh], "pitcher_id", "n", sr)
                        .reindex(pd.MultiIndex.from_arrays([pid, prev])).to_numpy().astype(np.float64), nan=0.0)
    gm_ = np.nan_to_num(piv_lookup(global_mix_n[global_mix_n["shape"] == sh], None, "n", sr).reindex(sr).to_numpy()
                        if False else np.zeros(1), nan=0.0)  # skip fine global norm, use freq only
    anchor = pp  # prior 실력으로 축소
    ctrl_t = (cs_ + K_CTRL * anchor) / (cn_ + K_CTRL)
    weight = fn_ + 1.0  # 최소 1로 스무딩
    num_pred += weight * ctrl_t
    den_mix += weight
    tot_n += cn_

pred = np.divide(num_pred, den_mix, out=pp.copy(), where=den_mix > 0)
dshape2 = pd.DataFrame({"shape_dev": pred - pp, "shape_n": np.log1p(tot_n)}, index=df.index)
print(f"  shape_dev(주변화) SD={dshape2['shape_dev'].std():.5f}", flush=True)

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
zdf = dshape2.loc[va]
keep = [c for c in zdf.columns if zdf[c].std() > 1e-12]
Z = zdf[keep].to_numpy(np.float64)
Zp = Z - Xd @ np.linalg.solve(XtX, Xd.T @ Z)
proj = Zp @ np.linalg.solve(Zp.T @ Zp + 1e-8 * np.eye(Zp.shape[1]), Zp.T @ e)
joint = (proj @ proj) / len(e) / bv * 1e5
mark = "채택후보" if joint >= 5 else ("애매" if joint >= 2 else "기각")
print(f"\n[pitch-shape 주변화] v15잔차 대비 합동가치 = {joint:+.1f}  ({mark})", flush=True)
for c in keep:
    z = zdf[c].to_numpy(np.float64)
    zp = z - Xd @ np.linalg.solve(XtX, Xd.T @ z)
    v = zp.var()
    if v < 1e-14:
        continue
    print(f"  {c:12s} corr={np.corrcoef(e, zp)[0,1]:+.5f}  개별={(np.cov(e,zp)[0,1]**2/v)/bv*1e5:+6.2f}", flush=True)
print(f"\n총 {time.time()-t0:.0f}s", flush=True)
