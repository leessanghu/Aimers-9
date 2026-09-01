"""Exact fold-A PCA audit for the five v60a members."""
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

avg = lambda paths: np.mean([np.load(p) for p in paths], axis=0)
base = avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6", "d8", "sub")])
hurdle = np.mean([
    (1 - np.load(f"phase90_cache/A_core_{n}.npy"))
    * np.load(f"phase90_cache/A_snc_{n}.npy")
    for n in ("d6", "d8")
], axis=0)
multires = avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42, 7)])
ordinal = avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42, 7)])
midother = avg([f"idea46_cache/A_midother_s{s}.npy" for s in (42, 7)])

names = ["base", "hurdle", "multires", "ordinal", "midother"]
members = np.column_stack([base, hurdle, multires, ordinal, midother])
z = (members - members.mean(0)) / members.std(0)
pca = PCA().fit(z)
pcs = pca.transform(z)

meta = pd.read_parquet("featcache_meta.parquet", columns=["season", "control_success"])
va = meta["season"].to_numpy() == 2024
y = meta.loc[va, "control_success"].to_numpy(float)
X = pd.read_parquet(
    "featcache_X.parquet",
    columns=["x_ability_here", "inseason_success_smooth", "inseason_cmd_index"],
).loc[va]
v60a = .24*base + .32*hurdle + .08*multires + .16*ordinal + .20*midother
bs = y.mean() * (1-y.mean())
score = 1e5 * (1 - np.mean((np.clip(v60a, 0, 1)-y)**2)/bs)

print(f"v60a local={score:.3f}")
print("PCA explained:", " ".join(f"PC{i+1}={v*100:.3f}%" for i,v in enumerate(pca.explained_variance_ratio_)))
print("member correlation")
print(pd.DataFrame(np.corrcoef(members, rowvar=False), index=names, columns=names).round(5))
for i in range(3):
    print(f"PC{i+1} feature corr:", {c: round(float(np.corrcoef(pcs[:,i], X[c])[0,1]), 5) for c in X})
print("v60a feature corr:", {c: round(float(np.corrcoef(v60a, X[c])[0,1]), 5) for c in X})
