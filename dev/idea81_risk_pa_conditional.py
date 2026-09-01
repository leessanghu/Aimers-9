"""Forward-validation screen for split risk correction and conditional PA-event aux heads.

The experiment deliberately separates calibration corrections from new information axes:
  1. Risk directions are mean-neutral and evaluated without fitting their magnitude.
  2. PA-event heads use fixed blend weights on 2022/2024 forward folds.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "idea81_cache"
CACHE.mkdir(exist_ok=True)
SEEDS = (42, 7)
FOLDS = (("C", 2021, 2022), ("A", 2023, 2024))
SUMMARY_ONLY = os.environ.get("IDEA81_SUMMARY_ONLY", "0") == "1"
t0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {message}", flush=True)


def avg(paths: list[Path]) -> np.ndarray:
    return np.mean([np.load(path) for path in paths], axis=0)


def score(y: np.ndarray, pred: np.ndarray) -> float:
    ref = float(y.mean() * (1.0 - y.mean()))
    return 1e5 * (1.0 - np.mean((np.clip(pred, 0.0, 1.0) - y) ** 2) / ref)


def baseline(tag: str, year: int, y: np.ndarray) -> tuple[np.ndarray, str]:
    base = avg([ROOT / "phase90_cache" / f"{tag}_base_{name}.npy"
                for name in ("d6", "d8", "sub")])
    hurdle = np.mean([
        (1.0 - np.load(ROOT / "phase90_cache" / f"{tag}_core_{name}.npy"))
        * np.load(ROOT / "phase90_cache" / f"{tag}_snc_{name}.npy")
        for name in ("d6", "d8")
    ], axis=0)
    mr = avg([ROOT / "idea13_cache" / f"{tag}_multires_s{s}.npy" for s in SEEDS])
    ordinal = avg([ROOT / "idea13_cache" / f"{tag}_ordinal_s{s}.npy" for s in SEEDS])
    midother = avg([ROOT / "idea46_cache" / f"{tag}_midother_s{s}.npy" for s in SEEDS])
    condball = avg([ROOT / "idea54_cache" / f"{tag}_cond_ball_s{s}.npy" for s in SEEDS])
    countresid = avg([ROOT / "idea54_cache" / f"{tag}_count_resid_s{s}.npy" for s in SEEDS])
    future50 = avg([ROOT / "idea54_cache" / f"{tag}_future50_multi_s{s}.npy" for s in SEEDS])

    v66 = (.1824 * base + .2432 * hurdle + .0608 * mr + .1216 * ordinal
           + .1520 * midother + .0800 * condball + .0800 * countresid + .0800 * future50)
    if tag != "A":
        return v66, "v66 proxy"

    # Fold A has honest MC11 and InGame predictions, so it can reproduce the v82 structure.
    proba11 = np.load(ROOT / "idea75_cache" / "A_proba11.npy")
    cls11 = np.load(ROOT / "idea75_cache" / "A_cls11_valid.npy")
    train_mask = META["season"].to_numpy() <= 2023
    cls_all = CLS11
    succ = np.array([
        Y[train_mask & (cls_all == c)].mean() if np.any(train_mask & (cls_all == c)) else 0.0
        for c in range(11)
    ])
    mc11 = proba11 @ succ
    ingame = np.load(ROOT / "idea80_cache" / "A_ingame_heads_s42.npy")[:, 0]
    v82 = (.1426368 * base + .1901824 * hurdle + .0475456 * mr + .0950912 * ordinal
           + .1188640 * midother + .0625600 * condball + .0625600 * countresid
           + .0625600 * future50 + .1380000 * mc11 + .0800000 * ingame)
    assert len(v82) == len(y) == len(cls11)
    return v82, "v82 local proxy"


def split_masks(meta_val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    order = meta_val["row_num"].to_numpy()
    cut = np.median(order)
    return order <= cut, order > cut


def fixed_direction_report(name: str, y: np.ndarray, base: np.ndarray,
                           raw_direction: np.ndarray, target_sd: float) -> dict[str, float]:
    direction = raw_direction - raw_direction.mean()
    if direction.std() == 0:
        raise ValueError(f"zero-variance direction: {name}")
    correction = direction * (target_sd / direction.std())
    early, late = split_masks(META.loc[META["season"] == 2024])
    full_delta = score(y, base + correction) - score(y, base)
    early_delta = score(y[early], (base + correction)[early]) - score(y[early], base[early])
    late_delta = score(y[late], (base + correction)[late]) - score(y[late], base[late])
    denom = float(np.mean(direction ** 2))
    oracle_alpha = -float(np.mean(direction * (base - y))) / denom
    return {
        "candidate": name,
        "correction_mean": float(correction.mean()),
        "correction_sd": float(correction.std()),
        "delta_full": full_delta,
        "delta_early": early_delta,
        "delta_late": late_delta,
        "oracle_alpha_raw": oracle_alpha,
        "corr_with_residual": float(np.corrcoef(direction, y - base)[0, 1]),
    }


def cluster_bootstrap_delta(y: np.ndarray, base: np.ndarray, candidate: np.ndarray,
                            pitcher: np.ndarray, n_boot: int = 400) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260825)
    ids = np.unique(pitcher)
    groups = {pid: np.flatnonzero(pitcher == pid) for pid in ids}
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([groups[pid] for pid in sampled])
        deltas[b] = score(y[idx], candidate[idx]) - score(y[idx], base[idx])
    return tuple(np.quantile(deltas, (0.10, 0.50, 0.90)))


log("load caches and labels")
X = pd.read_parquet(ROOT / "featcache_X.parquet")
META = pd.read_parquet(ROOT / "featcache_meta.parquet")
Y = META["control_success"].to_numpy(np.float64)
SEASON = META["season"].to_numpy(np.int64)
CLS5 = np.load(ROOT / "cls5_labels.npy")
PT = np.load(ROOT / "pitchtype_labels.npy")
PA_EVENT = np.load(ROOT / "paevent_labels.npy")
valid11 = (CLS5 >= 0) & (PT >= 0)
CLS11 = np.full(len(CLS5), -1, dtype=np.int64)
nd = valid11 & (CLS5 >= 2)
CLS11[nd] = (CLS5[nd] - 2) * 3 + PT[nd]
CLS11[valid11 & (CLS5 == 0)] = 9
CLS11[valid11 & (CLS5 == 1)] = 10


# ---------------------------------------------------------------------------
# Priority 1: middle/reverse risk directions on the honest fold-A MC11 output.
# ---------------------------------------------------------------------------
log("priority 1: split-risk direction screen")
mask_a = SEASON == 2024
y_a = Y[mask_a]
meta_a = META.loc[mask_a]
base_a, base_name = baseline("A", 2024, y_a)
P11 = np.load(ROOT / "idea75_cache" / "A_proba11.npy")
p_mid, p_rev = P11[:, 9], P11[:, 10]
total_cut = np.maximum(0.0, p_mid + p_rev - 0.25)
target_sd = float((0.045 * (total_cut - total_cut.mean())).std())
risk_directions = {
    "total_hinge_current": -total_cut,
    "middle_linear": -p_mid,
    "reverse_linear": -p_rev,
    "middle_hinge_010": -np.maximum(0.0, p_mid - 0.10),
    "reverse_hinge_015": -np.maximum(0.0, p_rev - 0.15),
    "split_hinges_equal": -(np.maximum(0.0, p_mid - 0.10) + np.maximum(0.0, p_rev - 0.15)),
}
risk_rows = []
for name, direction in risk_directions.items():
    row = fixed_direction_report(name, y_a, base_a, direction, target_sd)
    centered = direction - direction.mean()
    correction = centered * target_sd / centered.std()
    q10, q50, q90 = cluster_bootstrap_delta(
        y_a, base_a, base_a + correction, meta_a["pitcher_id"].to_numpy()
    )
    row.update(boot_q10=q10, boot_q50=q50, boot_q90=q90)
    risk_rows.append(row)
    log(f"  {name:22s} full={row['delta_full']:+7.3f} early={row['delta_early']:+7.3f} "
        f"late={row['delta_late']:+7.3f} boot80=[{q10:+.2f},{q90:+.2f}]")
pd.DataFrame(risk_rows).to_csv(CACHE / "risk_results.csv", index=False, encoding="utf-8-sig")


# ---------------------------------------------------------------------------
# Priority 2: shared-tree y + conditional PA-event heads on forward folds.
# ---------------------------------------------------------------------------
log("priority 2: conditional PA-event shared-tree heads")
nd_strike = CLS5 == 3
head_not_end = np.where(nd_strike & (PA_EVENT >= 0), (PA_EVENT != 3).astype(float), np.nan)
head_continue_strike = np.where(nd_strike & (PA_EVENT >= 0), (PA_EVENT == 1).astype(float), np.nan)
TARGETS = np.column_stack([Y, head_not_end, head_continue_strike])
log(f"  nd&strike coverage={np.mean(nd_strike):.2%}, not_end={np.nanmean(head_not_end):.3f}, "
    f"continue_strike={np.nanmean(head_continue_strike):.3f}")

pa_rows = []
for tag, upto, val_year in FOLDS:
    tr = SEASON <= upto
    va = SEASON == val_year
    yv = Y[va]
    meta_v = META.loc[va]
    base_v, fold_base_name = baseline(tag, val_year, yv)
    recency = 0.5 ** ((upto - SEASON) / 2.0)
    train_idx = np.flatnonzero(tr)
    cut = int(len(train_idx) * 0.92)
    fit_idx, es_idx = train_idx[:cut], train_idx[cut:]
    members: list[tuple[int, np.ndarray]] = []
    for seed in SEEDS:
        cache = CACHE / f"{tag}_pa_cond_s{seed}.npy"
        if cache.exists():
            pred = np.load(cache)
            log(f"  {tag}/s{seed}: cached")
        elif SUMMARY_ONLY:
            log(f"  {tag}/s{seed}: missing, skipped in summary-only mode")
            continue
        else:
            model = CatBoostRegressor(
                iterations=500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                loss_function="MultiRMSEWithMissingValues", random_seed=seed,
                early_stopping_rounds=50, verbose=100,
                thread_count=max(1, (os.cpu_count() or 4) - 1),
            )
            started = time.time()
            model.fit(X.iloc[fit_idx], TARGETS[fit_idx], sample_weight=recency[fit_idx],
                      eval_set=(X.iloc[es_idx], TARGETS[es_idx]))
            pred = np.clip(model.predict(X.loc[va])[:, 0], 0.0, 1.0)
            np.save(cache, pred)
            log(f"  {tag}/s{seed}: best_iter={model.best_iteration_}, elapsed={time.time()-started:.0f}s")
        members.append((seed, pred))

    early, late = split_masks(meta_v)
    report_members = [(f"seed{seed}", pred) for seed, pred in members]
    if len(members) >= 2:
        report_members.append(("seed_avg", np.mean([pred for _, pred in members], axis=0)))
    for member_name, pred in report_members:
        for weight in (0.05, 0.08, 0.10):
            blended = (1.0 - weight) * base_v + weight * pred
            delta = score(yv, blended) - score(yv, base_v)
            de = score(yv[early], blended[early]) - score(yv[early], base_v[early])
            dl = score(yv[late], blended[late]) - score(yv[late], base_v[late])
            q10, q50, q90 = cluster_bootstrap_delta(
                yv, base_v, blended, meta_v["pitcher_id"].to_numpy(), n_boot=300
            )
            row = {
                "fold": tag, "valid_year": val_year, "baseline": fold_base_name,
                "member": member_name, "weight": weight, "solo_score": score(yv, pred),
                "delta_full": delta, "delta_early": de, "delta_late": dl,
                "pred_corr_base": float(np.corrcoef(pred, base_v)[0, 1]),
                "error_corr_base": float(np.corrcoef(pred - yv, base_v - yv)[0, 1]),
                "mean_change": float((blended - base_v).mean()),
                "boot_q10": q10, "boot_q50": q50, "boot_q90": q90,
            }
            pa_rows.append(row)
            if member_name == "seed_avg":
                log(f"  {tag} avg w={weight:.2f}: full={delta:+7.3f} early={de:+7.3f} "
                    f"late={dl:+7.3f} boot80=[{q10:+.2f},{q90:+.2f}]")
    pd.DataFrame(pa_rows).to_csv(
        CACHE / "pa_conditional_results.csv", index=False, encoding="utf-8-sig"
    )

pa_df = pd.DataFrame(pa_rows)
pa_df.to_csv(CACHE / "pa_conditional_results.csv", index=False, encoding="utf-8-sig")


print("\n" + "=" * 92)
print("FINAL SCREEN")
print("=" * 92)
risk_df = pd.DataFrame(risk_rows).sort_values("delta_full", ascending=False)
print("\n[Priority 1: same-risk-SD, mean-neutral]")
print(risk_df[["candidate", "delta_full", "delta_early", "delta_late",
               "boot_q10", "boot_q90", "corr_with_residual"]].to_string(index=False))
print("\n[Priority 2: seed-average fixed weights]")
summary = pa_df[(pa_df.member == "seed_avg") | ((pa_df.fold == "A") & (pa_df.member == "seed42"))]
print(summary[["fold", "member", "weight", "solo_score", "delta_full",
      "delta_early", "delta_late", "boot_q10", "boot_q90", "pred_corr_base",
      "error_corr_base", "mean_change"]].to_string(index=False))
log("done")
