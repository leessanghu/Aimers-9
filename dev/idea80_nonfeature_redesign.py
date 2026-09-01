"""Four non-feature redesign experiments on the fold-A v78 local proxy.

1. Decode the unused InGame auxiliary head as a signed correction.
2. Anchor pitcher-season skill and learn a group-centered context residual.
3. Use a grouped Binomial / empirical-Bayes estimator only against v78.
4. Replace global weights with a low-capacity reliability gate.

No submission artifact is created by this script.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy.optimize import minimize
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "idea80_cache"
CACHE.mkdir(exist_ok=True)
T0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def load_avg(paths: list[str]) -> np.ndarray:
    return np.mean([np.load(ROOT / p) for p in paths], axis=0)


def score_factory(yv: np.ndarray):
    rate = float(yv.mean())
    bs = rate * (1.0 - rate)

    def score(p: np.ndarray, mask: np.ndarray | None = None) -> float:
        yy = yv if mask is None else yv[mask]
        pp = p if mask is None else p[mask]
        # Keep the official fold-wide uncertainty for subset comparisons so
        # early/late deltas remain on one common scale.
        return 1e5 * (1.0 - np.mean((np.clip(pp, 0.0, 1.0) - yy) ** 2) / bs)

    return score


def best_alpha(base: np.ndarray, residual: np.ndarray, y: np.ndarray, fit: np.ndarray) -> float:
    den = float(np.mean(residual[fit] ** 2))
    if den < 1e-14:
        return 0.0
    return float(np.mean((y[fit] - base[fit]) * residual[fit]) / den)


def report_signed(name: str, base: np.ndarray, residual: np.ndarray, yv: np.ndarray,
                  early: np.ndarray, late: np.ndarray, score) -> dict[str, float]:
    alpha = float(np.clip(best_alpha(base, residual, yv, early), -5.0, 5.0))
    full_alpha = float(np.clip(best_alpha(base, residual, yv, np.ones(len(yv), bool)), -5.0, 5.0))
    p = np.clip(base + alpha * residual, 0.0, 1.0)
    p_full = np.clip(base + full_alpha * residual, 0.0, 1.0)
    row = {
        "experiment": name,
        "alpha_early": alpha,
        "delta_early": score(p, early) - score(base, early),
        "delta_late": score(p, late) - score(base, late),
        "alpha_full_oracle": full_alpha,
        "delta_full_fixed": score(p) - score(base),
        "delta_full_oracle": score(p_full) - score(base),
        "resid_sd": float(np.std(residual)),
        "corr_base": float(np.corrcoef(residual, base)[0, 1]),
        "corr_innovation": float(np.corrcoef(residual, yv - base)[0, 1]),
    }
    log(f"{name}: a(early)={alpha:+.4f} early={row['delta_early']:+.3f} "
        f"late={row['delta_late']:+.3f} full={row['delta_full_fixed']:+.3f}; "
        f"oracle a={full_alpha:+.4f} d={row['delta_full_oracle']:+.3f}")
    return row


log("load feature/meta caches")
X = pd.read_parquet(ROOT / "featcache_X.parquet")
meta = pd.read_parquet(ROOT / "featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
season = meta["season"].to_numpy(np.int16)
row_num = meta["row_num"].to_numpy(np.int64)
pid = meta["pitcher_id"].to_numpy()
tr = season <= 2023
va = season == 2024
yv = y[va]
score = score_factory(yv)

# A chronological split inside the untouched 2024 validation season. It is
# used only to choose low-dimensional correction/gate coefficients; reported
# late deltas are therefore honest with respect to those coefficients.
rv = row_num[va]
cut = np.quantile(rv, 0.40)
early = rv <= cut
late = ~early
log(f"fold A train={tr.sum():,} valid={va.sum():,}; coefficient fit/eval={early.sum():,}/{late.sum():,}")

base = load_avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6", "d8", "sub")])
hurdle = np.mean([
    (1.0 - np.load(ROOT / f"phase90_cache/A_core_{n}.npy"))
    * np.load(ROOT / f"phase90_cache/A_snc_{n}.npy") for n in ("d6", "d8")
], axis=0)
mr = load_avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42, 7)])
ordinal = load_avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42, 7)])
midother = load_avg([f"idea46_cache/A_midother_s{s}.npy" for s in (42, 7)])
condball = load_avg([f"idea54_cache/A_cond_ball_s{s}.npy" for s in (42, 7)])
countresid = load_avg([f"idea54_cache/A_count_resid_s{s}.npy" for s in (42, 7)])
future50 = load_avg([f"idea54_cache/A_future50_multi_s{s}.npy" for s in (42, 7)])
v66 = (0.1824 * base + 0.2432 * hurdle + 0.0608 * mr + 0.1216 * ordinal
       + 0.1520 * midother + 0.08 * condball + 0.08 * countresid + 0.08 * future50)
mc5 = np.load(ROOT / "idea72_cache/A_mc.npy")
v78 = 0.85 * v66 + 0.15 * mc5
log(f"local baselines: v66={score(v66):.3f} v78proxy={score(v78):.3f} mc5={score(mc5):.3f}")


# ---------------------------------------------------------------------------
# Reconstruct legal current-season sufficient statistics.
# ---------------------------------------------------------------------------
log("reconstruct current-season success counts")
raw = pd.read_csv(
    ROOT.parent / "data/train.csv", encoding="utf-8-sig",
    usecols=["row_id", "pitcher_id", "season", "asof_pitcher_n",
             "asof_pitcher_success_rate", "control_success"],
)
raw["row_num"] = raw["row_id"].str.replace("TRAIN_", "", regex=False).astype(np.int64)
ordered = raw.sort_values(["pitcher_id", "row_num"])
last = ordered.groupby(["pitcher_id", "season"], as_index=False).last()
nb = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
last["N_end"] = nb + 1.0
last["S_end"] = (np.round(last["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * nb)
                 + last["control_success"].to_numpy(np.float64))
seasons = sorted(raw["season"].unique())
pn = last.pivot(index="pitcher_id", columns="season", values="N_end").reindex(columns=seasons).ffill(axis=1)
ps = last.pivot(index="pitcher_id", columns="season", values="S_end").reindex(columns=seasons).ffill(axis=1)
idx_prev = pd.MultiIndex.from_arrays([raw["pitcher_id"], raw["season"] - 1])
pn_s = pn.stack(future_stack=True)
ps_s = ps.stack(future_stack=True)
n_prev = np.nan_to_num(pn_s.reindex(idx_prev).to_numpy(np.float64), nan=0.0)
s_prev = np.nan_to_num(ps_s.reindex(idx_prev).to_numpy(np.float64), nan=0.0)
n_now = raw["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
s_now = np.round(raw["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now)
n_season = np.clip(n_now - n_prev, 0.0, None)
s_season = np.clip(s_now - s_prev, 0.0, None)
global_rate = float(y[tr].mean())
prior = np.divide(s_prev, n_prev, out=np.full(len(raw), global_rate), where=n_prev > 0)


# Choose empirical-Bayes strength on 2023 only, then freeze for 2024.
ks = np.array([2, 5, 10, 15, 20, 30, 50, 80, 120, 200, 400], np.float64)
sel23 = season == 2023
r23 = float(y[sel23].mean())
bs23 = r23 * (1.0 - r23)
eb_rows = []
for k in ks:
    theta = (s_season + k * prior) / (n_season + k)
    s23 = 1e5 * (1.0 - np.mean((theta[sel23] - y[sel23]) ** 2) / bs23)
    eb_rows.append((k, s23))
best_k = float(max(eb_rows, key=lambda z: z[1])[0])
theta_all = (s_season + best_k * prior) / (n_season + best_k)
theta = theta_all[va]
log(f"EB K selected on 2023: K={best_k:g}; theta2024 score={score(theta):.3f}")


# ---------------------------------------------------------------------------
# 1. Train the production-like InGame two-head model and decode head1.
# ---------------------------------------------------------------------------
ing_cache = CACHE / "A_ingame_heads_s42.npy"
es_cache = CACHE / "A_ingame_es_heads_s42.npy"
es_idx_cache = CACHE / "A_ingame_es_idx_s42.npy"
if ing_cache.exists() and es_cache.exists() and es_idx_cache.exists():
    heads_ing = np.load(ing_cache)
    heads_es = np.load(es_cache)
    es_idx = np.load(es_idx_cache)
    log("load cached InGame fold predictions")
else:
    ingame = np.load(ROOT / "ingame_rate.npy")
    ok = tr & np.isfinite(ingame)
    idx = np.flatnonzero(ok)
    nfit = int(len(idx) * 0.92)
    fit_idx, es_idx = idx[:nfit], idx[nfit:]
    Y = np.column_stack([y, ingame])
    w = 0.5 ** ((2023.0 - season) / 2.0)
    model = CatBoostRegressor(
        iterations=700, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
        loss_function="MultiRMSEWithMissingValues", random_seed=42,
        early_stopping_rounds=50, verbose=100, thread_count=4,
    )
    log("train fold-A InGame two-head CatBoost")
    model.fit(X.iloc[fit_idx], Y[fit_idx], sample_weight=w[fit_idx],
              eval_set=(X.iloc[es_idx], Y[es_idx]))
    heads_ing = np.clip(model.predict(X.loc[va]), 0.0, 1.0)
    heads_es = np.clip(model.predict(X.iloc[es_idx]), 0.0, 1.0)
    np.save(ing_cache, heads_ing)
    np.save(es_cache, heads_es)
    np.save(es_idx_cache, es_idx)
    log(f"InGame trained best_iter={model.best_iteration_}")

h0, h1 = heads_ing[:, 0], heads_ing[:, 1]
results: list[dict[str, float]] = []
results.append(report_signed("1a_head0_member", v78, h0 - v78, yv, early, late, score))

# Train-only nuisance mapping for h1. No validation labels or cross-row test
# information enters this mapping.
nuisance_cols = [
    "x_ability_here", "inseason_success_smooth", "inseason_cmd_index",
    "asof_pitcher_success_rate_smooth", "inseason_n", "role_n_app",
    "cat_game_type", "game_month",
]
nuisance_cols = [c for c in nuisance_cols if c in X.columns]
scaler = StandardScaler().fit(X.iloc[es_idx][nuisance_cols])
ridge = Ridge(alpha=20.0).fit(scaler.transform(X.iloc[es_idx][nuisance_cols]), heads_es[:, 1])
h1_expected = ridge.predict(scaler.transform(X.loc[va, nuisance_cols]))
residuals = {
    "1b_h1_minus_h0": h1 - h0,
    "1c_h1_minus_ability": h1 - X.loc[va, "inseason_success_smooth"].to_numpy(np.float64),
    "1d_h1_nuisance_resid": h1 - h1_expected,
    "1e_h1_minus_v78": h1 - v78,
}
for name, resid in residuals.items():
    resid = resid - float(np.mean(resid[early]))
    results.append(report_signed(name, v78, resid, yv, early, late, score))


# ---------------------------------------------------------------------------
# 2. Anchored backfitting: group-center y-theta, then learn only context.
# ---------------------------------------------------------------------------
anchor_cache = CACHE / "A_anchor_context_resid_s42.npy"
if anchor_cache.exists():
    anchor_resid = np.load(anchor_cache)
    log("load cached anchor residual")
else:
    usage = pd.read_csv(ROOT / "idea57_feature_usage_detail.csv")
    allowed = {"game_context", "batter_matchup", "environment_team", "trackman"}
    cols = usage.loc[usage["block"].isin(allowed), "feature"].tolist()
    cols += [
        "x_kal_minus_career", "x_prev5_minus_career", "x_prev1_minus_prev5",
        "ly_minus_career", "vol_std", "vol_range", "form_accel",
        "form_1_minus_3", "form_3_minus_5", "inseason_middle_minus_career",
    ]
    cols = [c for c in dict.fromkeys(cols) if c in X.columns]
    raw_resid = y - theta_all
    gdf = pd.DataFrame({"pid": pid, "season": season, "r": raw_resid})
    group_mean = gdf.groupby(["pid", "season"])["r"].transform("mean").to_numpy(np.float64)
    target = raw_resid - group_mean
    model = HistGradientBoostingRegressor(
        loss="squared_error", max_iter=250, learning_rate=0.03, max_depth=6,
        max_leaf_nodes=31, l2_regularization=10.0, early_stopping=True,
        validation_fraction=0.10, n_iter_no_change=25, random_state=42,
    )
    log(f"train anchored context residual HGB: rows={tr.sum():,} cols={len(cols)}")
    model.fit(X.loc[tr, cols], target[tr], sample_weight=0.5 ** ((2023 - season[tr]) / 2.0))
    anchor_resid = model.predict(X.loc[va, cols])
    np.save(anchor_cache, anchor_resid)
    log(f"anchor residual trained n_iter={model.n_iter_}")

results.append(report_signed("2a_theta_plus_context", theta, anchor_resid, yv, early, late, score))
results.append(report_signed("2b_v78_plus_anchor_context", v78, anchor_resid, yv, early, late, score))


# ---------------------------------------------------------------------------
# 3. Grouped Binomial / EB: requested v78-only comparison.
# ---------------------------------------------------------------------------
eb_delta = theta - v78
results.append(report_signed("3_v78_plus_EB_direction", v78, eb_delta, yv, early, late, score))
for w0 in (0.02, 0.05, 0.08, 0.10, 0.15, 0.20):
    p = (1.0 - w0) * v78 + w0 * theta
    log(f"3 fixed EB weight={w0:.2f}: full delta={score(p)-score(v78):+.3f} "
        f"late delta={score(p,late)-score(v78,late):+.3f}")


# ---------------------------------------------------------------------------
# 4. Low-capacity reliability gates, fitted early-2024 and tested late-2024.
# ---------------------------------------------------------------------------
z_n = np.log1p(n_season[va])
z_n = (z_n - z_n[early].mean()) / max(z_n[early].std(), 1e-8)
month = X.loc[va, "game_month"].to_numpy(np.float64)
z_m = (month - month[early].mean()) / max(month[early].std(), 1e-8)
first = (n_season[va] == 0).astype(np.float64)
G = np.column_stack([np.ones(len(yv)), z_n, z_m, first])


def fit_gate(other: np.ndarray, name: str) -> dict[str, float]:
    diff = other - v78

    def sigmoid(t):
        return 1.0 / (1.0 + np.exp(-np.clip(t, -30, 30)))

    def obj(beta):
        w = sigmoid(G[early] @ beta)
        p = v78[early] + w * diff[early]
        return float(np.mean((p - yv[early]) ** 2) + 1e-4 * np.sum(beta[1:] ** 2))

    opt = minimize(obj, np.array([-2.5, 0.0, 0.0, 0.0]), method="BFGS")
    wg = sigmoid(G @ opt.x)
    pg = v78 + wg * diff
    # Fair comparator: one constant weight fitted on exactly the same early rows.
    wc = float(np.clip(best_alpha(v78, diff, yv, early), 0.0, 1.0))
    pc = v78 + wc * diff
    row = {
        "experiment": f"4_gate_{name}",
        "alpha_early": float(wg.mean()),
        "delta_early": score(pg, early) - score(v78, early),
        "delta_late": score(pg, late) - score(v78, late),
        "alpha_full_oracle": wc,
        "delta_full_fixed": score(pg) - score(v78),
        "delta_full_oracle": score(pc) - score(v78),
        "resid_sd": float(wg.std()),
        "corr_base": float(np.corrcoef(wg, z_n)[0, 1]),
        "corr_innovation": float(score(pg, late) - score(pc, late)),
    }
    log(f"4 gate {name}: beta={np.round(opt.x,3)} mean/std(w)={wg.mean():.3f}/{wg.std():.3f}; "
        f"late gate={row['delta_late']:+.3f}, constant(w={wc:.3f})="
        f"{score(pc,late)-score(v78,late):+.3f}, gate-minus-constant={row['corr_innovation']:+.3f}")
    return row


results.append(fit_gate(theta, "EB"))
results.append(fit_gate(h0, "InGame_h0"))


out = pd.DataFrame(results)
out.to_csv(ROOT / "idea80_nonfeature_redesign_results.csv", index=False)
pd.DataFrame(eb_rows, columns=["K", "score_2023"]).to_csv(
    ROOT / "idea80_eb_k_results.csv", index=False)
print("\n" + "=" * 120)
print("NON-FEATURE REDESIGN SUMMARY (fold A v78 proxy)")
print("=" * 120)
print(out.to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
log("done; no packaging")
