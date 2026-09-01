"""idea55: conditional recovered-label taxonomy screen.

Evidence-driven candidates after v62/v63/v64 all beat their negative local score:
  danger_ball  = [y, 1-ball only among dangerous failures] (v62 complement)
  failure_ball = [y, 1-ball only among every y=0 row]       (broader failure type)
  ball_success = [y, y only among ball=1 rows]              (rotate contingency table)

Protocol intentionally matches idea54 fold A.  v62 cond_ball and v64 future50
cached predictions are reported as measured positive controls.  Stage 1 uses one
seed for all candidates; set RUN_SECOND_SEED=True only after reviewing diversity.
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea55_cache"
os.makedirs(CD, exist_ok=True)
SEEDS = [42, 7]
# Stage 2: only the stage-1 winner receives the second seed.
RUN_SECOND_SEED = True
VALIDATE_ONLY = "danger_ball"
t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time()-t0:5.0f}s] {msg}", flush=True)


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


log("load feature cache and recovered labels")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
season = meta["season"].to_numpy(np.float64)
rev = recover(meta, "asof_pitcher_reverse_rate")
mid = recover(meta, "asof_pitcher_middle_rate")
ball = recover(meta, "asof_pitcher_ball_rate")
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
danger = valid & ((rev > 0.5) | (mid > 0.5))

targets = {
    "danger_ball": np.where(danger, 1.0 - ball, np.nan),
    "failure_ball": np.where(valid & (y < 0.5), 1.0 - ball, np.nan),
    "ball_success": np.where(valid & (ball > 0.5), y, np.nan),
}
for name, target in targets.items():
    log(f"{name}: coverage={np.isfinite(target).mean():.2%}, mean={np.nanmean(target):.4f}")
if VALIDATE_ONLY:
    targets = {VALIDATE_ONLY: targets[VALIDATE_ONLY]}

tr = season <= 2023
va = season == 2024
yv = y[va]
rate = yv.mean()
bs = rate * (1.0 - rate)
score = lambda p: 1e5 * (1.0 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / bs)
recency = 0.5 ** ((2023.0 - season) / 2.0)

avg = lambda paths: np.mean([np.load(p) for p in paths], axis=0)
base = avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6", "d8", "sub")])
hurdle = np.mean([
    (1.0 - np.load(f"phase90_cache/A_core_{n}.npy")) * np.load(f"phase90_cache/A_snc_{n}.npy")
    for n in ("d6", "d8")
], axis=0)
mr = avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42, 7)])
ordinal = avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42, 7)])
v47 = 0.30 * base + 0.40 * hurdle + 0.10 * mr + 0.20 * ordinal
v47_score = score(v47)
log(f"v47 local={v47_score:.3f}")

# Current champion proxy.  The four old blocks already average their production
# members; auxiliary blocks use seed42 to match the single production fit.
midother_42 = np.load("idea46_cache/A_midother_s42.npy")
condball_42 = np.load("idea54_cache/A_cond_ball_s42.npy")
countresid_42 = np.load("idea54_cache/A_count_resid_s42.npy")
future50_42 = np.load("idea54_cache/A_future50_multi_s42.npy")
v66 = (
    0.1824 * base + 0.2432 * hurdle + 0.0608 * mr + 0.1216 * ordinal
    + 0.1520 * midother_42
    + 0.0800 * condball_42 + 0.0800 * countresid_42 + 0.0800 * future50_42
)
v66_score = score(v66)
log(f"v66 production-proxy local={v66_score:.3f}")

positive_controls = {}
for label, path in {
    "v62_cond_ball_s42": "idea54_cache/A_cond_ball_s42.npy",
    "v64_future50_s42": "idea54_cache/A_future50_multi_s42.npy",
}.items():
    if os.path.exists(path):
        p = np.load(path)
        positive_controls[label] = p
        log(f"control {label}: solo={score(p):.3f}, blend10 delta={score(.9*v47+.1*p)-v47_score:+.3f}")

train_idx = np.flatnonzero(tr)
cut = int(len(train_idx) * 0.92)
fit_idx, es_idx = train_idx[:cut], train_idx[cut:]
params = dict(
    iterations=350, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
    loss_function="MultiRMSEWithMissingValues", verbose=0,
)
preds = {}
seeds = SEEDS if RUN_SECOND_SEED else SEEDS[:1]
for name, aux in targets.items():
    members = []
    for seed in seeds:
        cache = os.path.join(CD, f"A_{name}_s{seed}.npy")
        if os.path.exists(cache):
            p = np.load(cache)
        else:
            started = time.time()
            model = CatBoostRegressor(**params, random_seed=seed)
            Y = np.column_stack([y, aux])
            model.fit(
                X.iloc[fit_idx], Y[fit_idx], sample_weight=recency[fit_idx],
                eval_set=(X.iloc[es_idx], Y[es_idx]),
            )
            p = np.clip(model.predict(X.loc[va]), 0.0, 1.0)[:, 0]
            np.save(cache, p)
            log(f"{name}/s{seed}: best_iter={model.best_iteration_}, "
                f"solo={score(p):.3f}, blend10={score(.9*v47+.1*p)-v47_score:+.3f}, "
                f"elapsed={time.time()-started:.0f}s")
        members.append(p)
    preds[name] = np.mean(members, axis=0)

print("\n" + "=" * 100)
print("idea55 conditional taxonomy result (fold A; known actual positive controls shown for context)")
print("=" * 100)
control_ref = positive_controls.get("v62_cond_ball_s42")
future_ref = positive_controls.get("v64_future50_s42")
for name, p in preds.items():
    spread = 0.0
    if len(seeds) > 1:
        q = [np.load(os.path.join(CD, f"A_{name}_s{s}.npy")) for s in seeds]
        spread = max(score(z) for z in q) - min(score(z) for z in q)
    corr_cb = np.corrcoef(p, control_ref)[0, 1] if control_ref is not None else np.nan
    corr_f5 = np.corrcoef(p, future_ref)[0, 1] if future_ref is not None else np.nan
    delta_v66 = score(.92 * v66 + .08 * p) - v66_score
    corr_v66 = np.corrcoef(p, v66)[0, 1]
    repl_cb = score(v66 + .08 * (p - condball_42)) - v66_score
    repl_cr = score(v66 + .08 * (p - countresid_42)) - v66_score
    repl_f5 = score(v66 + .08 * (p - future50_42)) - v66_score
    print(f"{name:<14} solo={score(p):8.3f}  v47_blend10={score(.9*v47+.1*p)-v47_score:+7.3f}  "
          f"v66_blend08={delta_v66:+7.3f}  seed_spread={spread:6.2f}  "
          f"corr(v66)={corr_v66:.5f} corr(condball)={corr_cb:.5f} corr(future50)={corr_f5:.5f}  "
          f"replace(cb/cr/f5)={repl_cb:+.3f}/{repl_cr:+.3f}/{repl_f5:+.3f}")

names = list(preds)
print("\ncandidate prediction correlations")
print(pd.DataFrame(np.corrcoef([preds[n] for n in names]), index=names, columns=names).round(5))
log("done")
