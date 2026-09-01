"""Build two post-v66 candidates.

v68: same three proven v62/v63/v64 axes, total weight 0.30 (0.10 each).
v69: v66 plus a new complementary conditional failure-subtype axis:
     head0=y, head1=1-ball only when dangerous=(reverse OR middle).

The second target is train-only.  Inference uses head0 and each test row alone.
"""
from __future__ import annotations

import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

OUT = "../submit/model"
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time()-t0:5.0f}s] {msg}", flush=True)


def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for key, value in list(vars(obj).items()):
            if type(value).__name__ in ("Generator", "BitGenerator", "RandomState"):
                setattr(obj, key, None)
            else:
                strip_rng(value, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            strip_rng(value, seen, depth + 1)


def recover(meta: pd.DataFrame, col: str) -> np.ndarray:
    pid = meta["pitcher_id"].to_numpy()
    n = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
    same_next = np.zeros(len(meta), dtype=bool)
    same_next[order[:-1]] = pid[order][1:] == pid[order][:-1]
    cumulative = np.round(meta[col].fillna(0).to_numpy(np.float64) * n)
    ordered = cumulative[order]
    delta = np.empty(len(meta), dtype=np.float64)
    delta[:-1] = ordered[1:] - ordered[:-1]
    delta[-1] = np.nan
    delta[~same_next[order]] = np.nan
    out = np.empty(len(meta), dtype=np.float64)
    out[order] = delta
    return out


def rescale_weights(artifact: dict, factor: float) -> None:
    for key in list(artifact):
        if key.endswith("_weight") or key == "base_weight":
            artifact[key] = float(artifact[key]) * factor


log("v66 artifact and feature cache load")
v66 = joblib.load(os.path.join(OUT, "model_artifacts_v66.pkl"))
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
assert list(X.columns) == list(v66["feature_order"]), "feature cache/order mismatch"

# v68: along the already measured v65 -> v66 direction.  No retraining.
v68 = dict(v66)
old_new_total = sum(float(v68[k]) for k in ("condball_weight", "countresid_weight", "future50_weight"))
old_core = 1.0 - old_new_total
new_total = 0.30
core_factor = (1.0 - new_total) / old_core
for key in list(v68):
    if (key.endswith("_weight") or key == "base_weight") and key not in {
        "condball_weight", "countresid_weight", "future50_weight"
    }:
        v68[key] = float(v68[key]) * core_factor
for key in ("condball_weight", "countresid_weight", "future50_weight"):
    v68[key] = new_total / 3.0
joblib.dump(v68, os.path.join(OUT, "model_artifacts_v68.pkl"))
log("v68 saved: proven fusion total=0.30 (0.10 x 3)")

# v69: model the complementary half of v62's conditional taxonomy.
y = meta["control_success"].to_numpy(np.float64)
rev = recover(meta, "asof_pitcher_reverse_rate")
mid = recover(meta, "asof_pitcher_middle_rate")
ball = recover(meta, "asof_pitcher_ball_rate")
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
dangerous = valid & ((rev > 0.5) | (mid > 0.5))
head_danger_notball = np.where(dangerous, 1.0 - ball, np.nan)
Y = np.column_stack([y, head_danger_notball])
log(f"danger-ball conditional coverage={np.isfinite(head_danger_notball).mean():.2%}, "
    f"not-ball rate={np.nanmean(head_danger_notball):.4f}")

seasons = meta["season"].to_numpy(np.float64)
weights = 0.5 ** ((seasons.max() - seasons) / 2.0)
cut = int(len(X) * 0.92)
model = CatBoostRegressor(
    iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
    random_seed=42, verbose=0, loss_function="MultiRMSEWithMissingValues",
    early_stopping_rounds=50,
)
model.fit(X.iloc[:cut], Y[:cut], sample_weight=weights[:cut], eval_set=(X.iloc[cut:], Y[cut:]))
log(f"dangerball trained: best_iter={model.best_iteration_}")
strip_rng(model)

v69 = dict(v66)
new_weight = 0.08
rescale_weights(v69, 1.0 - new_weight)
v69["dangerball_model"] = model
v69["dangerball_weight"] = new_weight
weight_sum = sum(float(v) for k, v in v69.items() if k.endswith("_weight") or k == "base_weight")
assert abs(weight_sum - 1.0) < 1e-9, weight_sum
joblib.dump(v69, os.path.join(OUT, "model_artifacts_v69.pkl"))
log(f"v69 saved: dangerball={new_weight:.2f}, weight_sum={weight_sum:.6f}")

