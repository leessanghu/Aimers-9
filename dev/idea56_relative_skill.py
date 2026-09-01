"""idea56: league-centered pitcher-season resolution head, fold-A pilot.

The old multires targets absolute pitcher-season rates.  Murphy decomposition
shows the remaining gap is resolution, while v48 caps global calibration gain at
~2.4 LB points.  Center auxiliary targets by each season x game_type league rate
so shared trees must explain relative pitcher skill rather than seasonal level.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

t0 = time.time()
cache = "idea56_cache/A_relative_skill_s42.npy"
os.makedirs(os.path.dirname(cache), exist_ok=True)


def log(msg):
    print(f"[{time.time()-t0:5.0f}s] {msg}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
season = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
gt = meta["game_type"].astype(str).to_numpy()
same_hand = X["same_hand"].to_numpy(np.float64)
tr = season <= 2023
va = season == 2024

# Build all targets row-wise, but the fitted model sees only <=2023 rows.
d = pd.DataFrame({"pid": pid, "season": season, "gt": gt, "sh": same_hand, "y": y})
league = d.groupby(["season", "gt"])["y"].agg(ls="sum", ln="count")
d = d.join(league, on=["season", "gt"])
g = float(y[tr].mean())
K_LG, K_PS = 100.0, 15.0
league_loo = ((d["ls"] - d["y"]) + K_LG * g) / ((d["ln"] - 1) + K_LG)

ps = d.groupby(["pid", "season"])["y"].agg(ps="sum", pn="count")
d = d.join(ps, on=["pid", "season"])
ps_abs = ((d["ps"] - d["y"]) + K_PS * league_loo) / ((d["pn"] - 1) + K_PS)

psh = d.groupby(["pid", "season", "sh"])["y"].agg(hs="sum", hn="count")
d = d.join(psh, on=["pid", "season", "sh"])
psh_abs = ((d["hs"] - d["y"]) + K_PS * ps_abs) / ((d["hn"] - 1) + K_PS)

h_rel_ps = np.clip(0.5 + ps_abs.to_numpy(np.float64) - league_loo.to_numpy(np.float64), 0, 1)
h_rel_psh = np.clip(0.5 + psh_abs.to_numpy(np.float64) - league_loo.to_numpy(np.float64), 0, 1)
Y = np.column_stack([y, h_rel_ps, h_rel_psh])
log(f"relative targets: ps sd={h_rel_ps[tr].std():.4f}, psh sd={h_rel_psh[tr].std():.4f}")

if os.path.exists(cache):
    p = np.load(cache)
else:
    train_idx = np.flatnonzero(tr)
    cut = int(len(train_idx) * 0.92)
    fit_idx, es_idx = train_idx[:cut], train_idx[cut:]
    w = 0.5 ** ((2023.0 - season) / 2.0)
    model = CatBoostRegressor(
        iterations=350, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
        loss_function="MultiRMSEWithMissingValues", random_seed=42, verbose=0,
    )
    model.fit(X.iloc[fit_idx], Y[fit_idx], sample_weight=w[fit_idx],
              eval_set=(X.iloc[es_idx], Y[es_idx]))
    p = np.clip(model.predict(X.loc[va]), 0, 1)[:, 0]
    np.save(cache, p)
    log(f"trained best_iter={model.best_iteration_}")

yv = y[va]
bs = yv.mean() * (1 - yv.mean())
score = lambda z: 1e5 * (1 - np.mean((np.clip(z, 0, 1) - yv) ** 2) / bs)
avg = lambda paths: np.mean([np.load(q) for q in paths], axis=0)
base = avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6", "d8", "sub")])
hur = np.mean([(1-np.load(f"phase90_cache/A_core_{n}.npy"))*np.load(f"phase90_cache/A_snc_{n}.npy")
               for n in ("d6", "d8")], axis=0)
mr = avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42, 7)])
od = avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42, 7)])
mo = np.load("idea46_cache/A_midother_s42.npy")
cb = np.load("idea54_cache/A_cond_ball_s42.npy")
cr = np.load("idea54_cache/A_count_resid_s42.npy")
f5 = np.load("idea54_cache/A_future50_multi_s42.npy")
v66 = .1824*base + .2432*hur + .0608*mr + .1216*od + .152*mo + .08*cb + .08*cr + .08*f5

print("\nidea56 relative-skill fold-A pilot")
print(f"solo={score(p):.3f}  v66={score(v66):.3f}")
print(f"add08={score(.92*v66+.08*p)-score(v66):+.3f}")
print(f"replace_multires={score(v66+.0608*(p-mr))-score(v66):+.3f}")
print(f"corr(v66)={np.corrcoef(p,v66)[0,1]:.5f} corr(multires)={np.corrcoef(p,mr)[0,1]:.5f} "
      f"corr(condball)={np.corrcoef(p,cb)[0,1]:.5f} corr(future50)={np.corrcoef(p,f5)[0,1]:.5f}")
print("small-weight add grid:")
for ww in (.01, .02, .03, .04, .06, .08):
    print(f"  w={ww:.2f}: {score((1-ww)*v66+ww*p)-score(v66):+.3f}")
print("partial multires replacement grid:")
for frac in (.25, .50, .75, 1.00):
    ww = .0608 * frac
    print(f"  replace={frac:.2f}: {score(v66+ww*(p-mr))-score(v66):+.3f}")
log("done")
