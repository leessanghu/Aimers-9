"""Screen extra batter-side axes on top of the current v82/v88 local proxy.

The goal is not to tune a local correction.  We test whether batter-specific
history provides an independent prediction axis after the current 162 features.

Feature groups:
  deployable_prior: previous-season batter outcome priors, batter-count success.
  deployable_context: interaction/reliability transforms from public as-of fields.
  diagnostic_current_reverse: current-season batter reverse/ball reconstructed from
      train labels only. This is deliberately marked non-deployable.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

import batter_split as bsplit


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "idea82_cache"
CACHE.mkdir(exist_ok=True)
SEEDS = (42,)
FOLDS = (("C", 2021, 2022), ("A", 2023, 2024))
WEIGHTS = (0.03, 0.05, 0.08, 0.10)
t0 = time.time()


def log(message: str) -> None:
    print(f"[{time.time() - t0:6.0f}s] {message}", flush=True)


def score(y: np.ndarray, pred: np.ndarray) -> float:
    pred = np.clip(pred, 0.0, 1.0)
    ref = float(y.mean() * (1.0 - y.mean()))
    return 1e5 * (1.0 - np.mean((pred - y) ** 2) / ref)


def avg(paths: list[Path]) -> np.ndarray:
    return np.mean([np.load(path) for path in paths], axis=0)


def baseline(tag: str, y: np.ndarray) -> tuple[np.ndarray, str]:
    base = avg([ROOT / "phase90_cache" / f"{tag}_base_{name}.npy"
                for name in ("d6", "d8", "sub")])
    hurdle = np.mean([
        (1.0 - np.load(ROOT / "phase90_cache" / f"{tag}_core_{name}.npy"))
        * np.load(ROOT / "phase90_cache" / f"{tag}_snc_{name}.npy")
        for name in ("d6", "d8")
    ], axis=0)
    mr = avg([ROOT / "idea13_cache" / f"{tag}_multires_s{s}.npy" for s in (42, 7)])
    ordinal = avg([ROOT / "idea13_cache" / f"{tag}_ordinal_s{s}.npy" for s in (42, 7)])
    midother = avg([ROOT / "idea46_cache" / f"{tag}_midother_s{s}.npy" for s in (42, 7)])
    condball = avg([ROOT / "idea54_cache" / f"{tag}_cond_ball_s{s}.npy" for s in (42, 7)])
    countresid = avg([ROOT / "idea54_cache" / f"{tag}_count_resid_s{s}.npy" for s in (42, 7)])
    future50 = avg([ROOT / "idea54_cache" / f"{tag}_future50_multi_s{s}.npy" for s in (42, 7)])

    v66 = (.1824 * base + .2432 * hurdle + .0608 * mr + .1216 * ordinal
           + .1520 * midother + .0800 * condball + .0800 * countresid + .0800 * future50)
    if tag != "A":
        return v66, "v66 proxy"

    proba11 = np.load(ROOT / "idea75_cache" / "A_proba11.npy")
    train_mask = META["season"].to_numpy() <= 2023
    succ = np.array([
        Y[train_mask & (CLS11 == c)].mean() if np.any(train_mask & (CLS11 == c)) else 0.0
        for c in range(11)
    ])
    mc11 = proba11 @ succ
    ingame = np.load(ROOT / "idea80_cache" / "A_ingame_heads_s42.npy")[:, 0]
    v82 = (.1426368 * base + .1901824 * hurdle + .0475456 * mr + .0950912 * ordinal
           + .1188640 * midother + .0625600 * condball + .0625600 * countresid
           + .0625600 * future50 + .1380000 * mc11 + .0800000 * ingame)

    p_mid, p_rev = proba11[:, 9], proba11[:, 10]
    risk = np.maximum(0.0, p_mid + p_rev - 0.25)
    v88 = v82 - 0.045 * (risk - risk.mean())
    return v88, "v88 local proxy"


def split_masks(meta_val: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    order = meta_val["row_num"].to_numpy()
    cut = np.median(order)
    return order <= cut, order > cut


def cluster_bootstrap_delta(y: np.ndarray, base: np.ndarray, candidate: np.ndarray,
                            pitcher: np.ndarray, n_boot: int = 250) -> tuple[float, float, float]:
    rng = np.random.default_rng(20260825)
    ids = np.unique(pitcher)
    groups = {pid: np.flatnonzero(pitcher == pid) for pid in ids}
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([groups[pid] for pid in sampled])
        deltas[b] = score(y[idx], candidate[idx]) - score(y[idx], base[idx])
    return tuple(np.quantile(deltas, (0.10, 0.50, 0.90)))


def recover_pitch_labels(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    n = d["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    order = d.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
    pid_o = d["pitcher_id"].to_numpy()[order]
    n_o = n[order]
    same_next = np.zeros(len(d), dtype=bool)
    same_next[order[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)

    def diff_rate(col: str) -> np.ndarray:
        c = np.round(d[col].fillna(0).to_numpy(np.float64) * n)
        out = np.full(len(d), np.nan)
        out[order[:-1]] = np.diff(c[order])
        out[~same_next] = np.nan
        return out

    dr = diff_rate("asof_pitcher_reverse_rate")
    dm = diff_rate("asof_pitcher_middle_rate")
    db = diff_rate("asof_pitcher_ball_rate")
    ds = diff_rate("asof_pitcher_strike_rate")
    d["_label_valid"] = np.isfinite(dr) & np.isfinite(dm) & np.isfinite(db) & np.isfinite(ds)
    d["_reverse_event"] = np.where(d["_label_valid"], (dr > 0.5).astype(float), np.nan)
    d["_middle_event"] = np.where(d["_label_valid"], (dm > 0.5).astype(float), np.nan)
    d["_ball_event"] = np.where(d["_label_valid"], (db > 0.5).astype(float), np.nan)
    d["_strike_event"] = np.where(d["_label_valid"], (ds > 0.5).astype(float), np.nan)
    d["_not_control"] = 1.0 - d["control_success"].astype(float)
    return d


def season_prior_table(df: pd.DataFrame, target: str, key_cols: list[str]) -> pd.DataFrame:
    cols = key_cols + ["season"]
    g = (df.dropna(subset=[target])
           .groupby(cols)[target].agg(s="sum", n="count").sort_index())
    return g.groupby(level=list(range(len(key_cols)))).cumsum().reset_index()


def lookup_prior(df: pd.DataFrame, table: pd.DataFrame, key_cols: list[str], target_name: str,
                 seasons_range: list[int], global_rate: float, k: float) -> pd.DataFrame:
    idx_arrays = [df[c] for c in key_cols] + [df["season"] - 1]
    idx = pd.MultiIndex.from_arrays(idx_arrays)
    piv_idx = key_cols

    def lk(col: str) -> np.ndarray:
        p = table.pivot_table(index=piv_idx, columns="season", values=col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        return np.nan_to_num(p.stack(future_stack=True).reindex(idx).to_numpy(np.float64), nan=0.0)

    s, n = lk("s"), lk("n")
    rate = (s + k * global_rate) / (n + k)
    prefix = f"bat_{target_name}_{'_'.join(key_cols).replace('batter_id_', '')}"
    out = pd.DataFrame(index=df.index)
    out[f"{prefix}_prior"] = rate
    out[f"{prefix}_n"] = np.log1p(n)
    return out.astype(np.float64)


def build_extra_features(df: pd.DataFrame, seasons_range: list[int]) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    labeled = recover_pitch_labels(df)
    g = {name: float(labeled[col].mean(skipna=True)) for name, col in {
        "reverse": "_reverse_event",
        "ball": "_ball_event",
        "middle": "_middle_event",
        "not_control": "_not_control",
        "success": "control_success",
    }.items()}

    deployable_parts = []
    for name, col, k in [
        ("reverse", "_reverse_event", 200.0),
        ("ball", "_ball_event", 200.0),
        ("middle", "_middle_event", 200.0),
        ("not_control", "_not_control", 80.0),
    ]:
        table = season_prior_table(labeled, col, ["batter_id"])
        deployable_parts.append(lookup_prior(df, table, ["batter_id"], name, seasons_range, g[name], k))

    bmarg = bsplit.build_batter_marginal(df)
    b_prior = bsplit.lookup_batter_prior(df, bmarg, seasons_range, g["success"])
    deployable_parts.append(
        bsplit.transform_bcount(df, bsplit.build_bcount_table(df), b_prior, seasons_range,
                                k=bsplit.K_BCOUNT).reset_index(drop=True)
    )

    out_public = pd.DataFrame(index=df.index)
    bat_n = df["asof_batter_n"].fillna(0).to_numpy(np.float64)
    bat_success = df["asof_batter_success_rate"].fillna(g["success"]).to_numpy(np.float64)
    bat_middle = df["asof_batter_middle_rate"].fillna(float(df["asof_batter_middle_rate"].mean())).to_numpy(np.float64)
    out_public["bat_asof_not_control"] = 1.0 - bat_success
    out_public["bat_asof_middle_x_n"] = bat_middle * np.log1p(bat_n)
    out_public["bat_asof_success_x_count_state"] = bat_success * (df["balls_before"] * 4 + df["strikes_before"]).to_numpy(np.float64)
    out_public["bat_asof_middle_x_same_hand"] = bat_middle * (df["pitcher_hand"] == df["batter_hand"]).to_numpy(np.float64)
    deployable_parts.append(out_public)

    # Diagnostic-only current-season reverse and ball features. They use the
    # recovered event label in train rows and cannot be produced for test rows.
    diag_parts = []
    for name, col, k in [("cur_reverse", "_reverse_event", 30.0), ("cur_ball", "_ball_event", 30.0)]:
        valid = labeled[col].notna()
        tmp = labeled.loc[valid, ["batter_id", "season", "row_num", col]].sort_values("row_num")
        gtab = tmp.groupby(["batter_id", "season"], sort=False)[col]
        tmp["_s_prior"] = gtab.cumsum() - tmp[col]
        tmp["_n_prior"] = gtab.cumcount()
        s_prior = tmp["_s_prior"].reindex(labeled.index).fillna(0.0).to_numpy(np.float64)
        n_prior = tmp["_n_prior"].reindex(labeled.index).fillna(0.0).to_numpy(np.float64)
        rate = (s_prior + k * g[name.replace("cur_", "")]) / (n_prior + k)
        diag_parts.append(pd.DataFrame({
            f"bat_{name}_inseason_oracle": rate,
            f"bat_{name}_inseason_n_oracle": np.log1p(n_prior),
        }, index=df.index))

    deploy = pd.concat(deployable_parts, axis=1).reset_index(drop=True)
    diag = pd.concat(diag_parts, axis=1).reset_index(drop=True)
    groups = {
        "deployable_prior": deploy.columns[:8].tolist() + ["bcount_diff", "bcount_n"],
        "deployable_context": out_public.columns.tolist(),
        "diagnostic_current_reverse": diag.columns.tolist(),
    }
    return pd.concat([deploy, diag], axis=1).astype(np.float64), groups


log("load feature cache, metadata, and labels")
X_BASE = pd.read_parquet(ROOT / "featcache_X.parquet").reset_index(drop=True)
META = pd.read_parquet(ROOT / "featcache_meta.parquet").reset_index(drop=True)
TRAIN = pd.read_csv(ROOT.parent / "data" / "train.csv", encoding="utf-8-sig")
TRAIN["row_num"] = TRAIN["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
Y = META["control_success"].to_numpy(np.float64)
SEASON = META["season"].to_numpy(np.int64)
CLS5 = np.load(ROOT / "cls5_labels.npy")
PT = np.load(ROOT / "pitchtype_labels.npy")
valid11 = (CLS5 >= 0) & (PT >= 0)
CLS11 = np.full(len(CLS5), -1, dtype=np.int64)
nd = valid11 & (CLS5 >= 2)
CLS11[nd] = (CLS5[nd] - 2) * 3 + PT[nd]
CLS11[valid11 & (CLS5 == 0)] = 9
CLS11[valid11 & (CLS5 == 1)] = 10

assert len(TRAIN) == len(X_BASE) == len(META)
seasons_range = sorted(TRAIN["season"].unique().tolist())

log("build extra batter features")
X_EXTRA, GROUPS = build_extra_features(TRAIN, seasons_range)
X_ALL_GROUPS = {
    "deployable_only": pd.concat([X_BASE, X_EXTRA[GROUPS["deployable_prior"] + GROUPS["deployable_context"]]], axis=1),
    "deployable_prior": pd.concat([X_BASE, X_EXTRA[GROUPS["deployable_prior"]]], axis=1),
    "deployable_context": pd.concat([X_BASE, X_EXTRA[GROUPS["deployable_context"]]], axis=1),
    "diagnostic_with_current_reverse": pd.concat([X_BASE, X_EXTRA], axis=1),
}

rows: list[dict[str, float | str | int]] = []
importance_rows: list[dict[str, float | str | int]] = []

for tag, upto, val_year in FOLDS:
    tr = SEASON <= upto
    va = SEASON == val_year
    yv = Y[va]
    meta_v = META.loc[va].reset_index(drop=True)
    base_v, base_name = baseline(tag, yv)
    early, late = split_masks(meta_v)
    recency = 0.5 ** ((upto - SEASON) / 2.0)
    train_idx = np.flatnonzero(tr)
    cut = int(len(train_idx) * 0.92)
    fit_idx, es_idx = train_idx[:cut], train_idx[cut:]
    base_score = score(yv, base_v)
    log(f"fold {tag}: train<={upto} -> {val_year}, base={base_name} score={base_score:.3f}")

    for group_name, X in X_ALL_GROUPS.items():
        for seed in SEEDS:
            cache = CACHE / f"{tag}_{group_name}_s{seed}.npy"
            model_path = CACHE / f"{tag}_{group_name}_s{seed}.cbm"
            if cache.exists() and model_path.exists():
                pred = np.load(cache)
                model = CatBoostRegressor()
                model.load_model(str(model_path))
                log(f"  {tag}/{group_name}/s{seed}: cached")
            else:
                model = CatBoostRegressor(
                    iterations=700,
                    learning_rate=0.035,
                    depth=6,
                    l2_leaf_reg=7.0,
                    loss_function="RMSE",
                    eval_metric="RMSE",
                    random_seed=seed,
                    early_stopping_rounds=60,
                    verbose=100,
                    thread_count=max(1, (os.cpu_count() or 4) - 1),
                )
                started = time.time()
                model.fit(
                    X.iloc[fit_idx], Y[fit_idx],
                    sample_weight=recency[fit_idx],
                    eval_set=(X.iloc[es_idx], Y[es_idx]),
                )
                pred = np.clip(model.predict(X.loc[va]), 0.0, 1.0)
                np.save(cache, pred)
                model.save_model(str(model_path))
                log(f"  {tag}/{group_name}/s{seed}: best_iter={model.best_iteration_}, {time.time()-started:.0f}s")

            for w in WEIGHTS:
                blended = (1.0 - w) * base_v + w * pred
                q10, q50, q90 = cluster_bootstrap_delta(
                    yv, base_v, blended, meta_v["pitcher_id"].to_numpy()
                )
                rows.append({
                    "fold": tag,
                    "valid_year": val_year,
                    "baseline": base_name,
                    "group": group_name,
                    "seed": seed,
                    "weight": w,
                    "solo_score": score(yv, pred),
                    "delta_full": score(yv, blended) - base_score,
                    "delta_early": score(yv[early], blended[early]) - score(yv[early], base_v[early]),
                    "delta_late": score(yv[late], blended[late]) - score(yv[late], base_v[late]),
                    "pred_corr_base": float(np.corrcoef(pred, base_v)[0, 1]),
                    "error_corr_base": float(np.corrcoef(pred - yv, base_v - yv)[0, 1]),
                    "mean_change": float((blended - base_v).mean()),
                    "boot_q10": q10,
                    "boot_q50": q50,
                    "boot_q90": q90,
                })

            fi = model.get_feature_importance(Pool(X.loc[va], yv), type="PredictionValuesChange")
            for feature, value in sorted(zip(X.columns, fi), key=lambda kv: kv[1], reverse=True)[:40]:
                importance_rows.append({
                    "fold": tag,
                    "group": group_name,
                    "seed": seed,
                    "feature": feature,
                    "importance": float(value),
                    "is_new_feature": feature in set(X_EXTRA.columns),
                })

            top_new = [
                (f, v) for f, v in sorted(zip(X.columns, fi), key=lambda kv: kv[1], reverse=True)
                if f in set(X_EXTRA.columns)
            ][:10]
            log("    top new: " + ", ".join(f"{f}={v:.3f}" for f, v in top_new))

pd.DataFrame(rows).to_csv(CACHE / "batter_axis_results.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(importance_rows).to_csv(CACHE / "batter_axis_importance.csv", index=False, encoding="utf-8-sig")

res = pd.DataFrame(rows)
print("\n" + "=" * 110)
print("BATTER AXIS SCREEN")
print("=" * 110)
print(res.sort_values(["fold", "delta_full"], ascending=[True, False])[
    ["fold", "group", "weight", "solo_score", "delta_full", "delta_early", "delta_late",
     "boot_q10", "boot_q90", "pred_corr_base", "error_corr_base", "mean_change"]
].to_string(index=False))

imp = pd.DataFrame(importance_rows)
print("\nTop new-feature importances:")
print(imp[imp["is_new_feature"]].sort_values(["fold", "group", "importance"],
                                             ascending=[True, True, False])
      .groupby(["fold", "group"]).head(12)
      [["fold", "group", "feature", "importance"]].to_string(index=False))

log(f"saved: {CACHE / 'batter_axis_results.csv'}")
log(f"saved: {CACHE / 'batter_axis_importance.csv'}")
