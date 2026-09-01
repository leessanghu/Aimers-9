"""idea58: learn a pitcher-ability-orthogonal residual correction.

Target: y - pitcher-season leave-one-out mean.
Inputs: explicitly exclude pitcher_ability and sample_reliability blocks.
Output is a signed correction, not another probability member:
    p_final = p_v60a + alpha * residual_prediction
This is the key structural difference from all previous auxiliary heads.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:5.0f}s] {msg}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
tag = os.environ.get("IDEA58_FOLD", "A").upper()
if tag not in {"A", "C"}:
    raise ValueError("IDEA58_FOLD must be A or C")
upto, valid_season = {"A": (2023, 2024), "C": (2021, 2022)}[tag]
usage = pd.read_csv("idea57_feature_usage_detail.csv")
allowed_blocks = {"game_context", "batter_matchup", "environment_team", "trackman"}
cols = usage.loc[usage["block"].isin(allowed_blocks), "feature"].tolist()
variant = os.environ.get("IDEA58_VARIANT", "context").lower()
if variant == "condition":
    # Pitcher-relative changes, not pitcher level. These were previously swept
    # into the broad pitcher_ability block and incorrectly excluded.
    cols += [
        "x_kal_minus_career", "x_prev5_minus_career", "x_prev1_minus_prev5",
        "ly_minus_career", "vol_std", "vol_range", "form_accel",
        "form_1_minus_3", "form_3_minus_5", "inseason_middle_minus_career",
    ]
elif variant != "context":
    raise ValueError("IDEA58_VARIANT must be context or condition")
cols = list(dict.fromkeys(cols))
cols = [c for c in cols if c in X.columns]
Xr = X[cols].astype(np.float64)

y = meta["control_success"].to_numpy(np.float64)
season = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
tr = season <= upto
va = season == valid_season
yv = y[va]
log(f"ability-level-free X={Xr.shape}, variant={variant}, blocks={sorted(allowed_blocks)}")

# Exact pitcher-season LOO ability for sufficiently observed cells.  No test
# rows or validation labels enter the fitted residual learner.
d = pd.DataFrame({"pid": pid, "season": season, "y": y})
ps = d.groupby(["pid", "season"])["y"].agg(s="sum", n="count")
d = d.join(ps, on=["pid", "season"])
ability_loo = (d["s"].to_numpy(np.float64) - y) / np.maximum(d["n"].to_numpy(np.float64) - 1, 1)
resid = y - ability_loo
fit_mask = tr & (d["n"].to_numpy() >= 20)
log(f"residual target coverage={fit_mask.mean():.2%}, mean={resid[fit_mask].mean():+.6f}, sd={resid[fit_mask].std():.4f}")

cache = f"idea58_cache/{tag}_residual_{variant}_s42.npy"
# Backward compatibility with the first context-only run.
old_cache = f"idea58_cache/{tag}_residual_s42.npy"
if variant == "context" and not os.path.exists(cache) and os.path.exists(old_cache):
    cache = old_cache
os.makedirs(os.path.dirname(cache), exist_ok=True)
if os.path.exists(cache):
    rhat = np.load(cache)
else:
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=350, learning_rate=0.03, max_depth=6,
        max_leaf_nodes=31, l2_regularization=10.0, early_stopping=True,
        validation_fraction=0.10, n_iter_no_change=25, random_state=42,
    )
    w = 0.5 ** ((upto - season[fit_mask]) / 2.0)
    model.fit(Xr.loc[fit_mask], resid[fit_mask], sample_weight=w)
    rhat = model.predict(Xr.loc[va])
    np.save(cache, rhat)
    log(f"trained n_iter={model.n_iter_}")

# Reconstruct the exact v60a fold-A proxy used in idea49.
avg = lambda paths: np.mean([np.load(q) for q in paths], axis=0)
base = avg([f"phase90_cache/{tag}_base_{n}.npy" for n in ("d6", "d8", "sub")])
hur = np.mean([(1-np.load(f"phase90_cache/{tag}_core_{n}.npy"))*np.load(f"phase90_cache/{tag}_snc_{n}.npy")
               for n in ("d6", "d8")], axis=0)
mr = avg([f"idea13_cache/{tag}_multires_s{s}.npy" for s in (42, 7)])
od = avg([f"idea13_cache/{tag}_ordinal_s{s}.npy" for s in (42, 7)])
mo = avg([f"idea46_cache/{tag}_midother_s{s}.npy" for s in (42, 7)])
members = np.column_stack([base, hur, mr, od, mo])
v60a = .24*base + .32*hur + .08*mr + .16*od + .20*mo

z = (members - members.mean(axis=0)) / members.std(axis=0)
pc1 = PCA(n_components=1).fit_transform(z)[:, 0]

# Remove any remaining linear/quadratic PC1 component using fixed coefficients
# learned only from official train validation predictions (no y involved).
A = np.column_stack([np.ones(len(pc1)), pc1, pc1**2])
rhat_perp = rhat - A @ np.linalg.lstsq(A, rhat, rcond=None)[0]

bs = yv.mean() * (1-yv.mean())
score = lambda p: 1e5*(1-np.mean((np.clip(p,0,1)-yv)**2)/bs)
b0 = score(v60a)

print(f"\nidea58 orthogonal residual fold-{tag} result ({upto}->{valid_season})")
print(f"v60a local={b0:.3f}" + (" (official measured 1080.5954)" if tag == "A" else ""))
print(f"rhat mean/sd={rhat.mean():+.6f}/{rhat.std():.6f}  corr(PC1)={np.corrcoef(rhat,pc1)[0,1]:+.5f}")
print(f"rperp mean/sd={rhat_perp.mean():+.6f}/{rhat_perp.std():.6f}  corr(PC1)={np.corrcoef(rhat_perp,pc1)[0,1]:+.5f}")
print("alpha grid: p = v60a + alpha * residual")
rows = []
for kind, rp in [("raw", rhat), ("perp", rhat_perp)]:
    for alpha in (-1.0, -.5, -.25, -.10, .10, .25, .50, 1.0):
        delta = score(v60a + alpha*rp) - b0
        rows.append((kind, alpha, delta))
        print(f"  {kind:<4} alpha={alpha:+.2f}: {delta:+.3f}")
    fine = np.linspace(-0.10, 0.40, 101)
    vals = np.array([score(v60a + a*rp)-b0 for a in fine])
    j = int(vals.argmax())
    print(f"  {kind:<4} fine optimum alpha={fine[j]:+.3f}: {vals[j]:+.3f}; "
          f"corr(target innovation)={np.corrcoef(rp, yv-v60a)[0,1]:+.5f}")
pd.DataFrame(rows, columns=["kind", "alpha", "delta"]).to_csv(
    f"idea58_{tag}_{variant}_alpha_grid.csv", index=False)
log("done")
