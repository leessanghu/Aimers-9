"""Phase 42: v15 91 features vs time-opened 103 features across 5 single models.

Goal:
  - Reproduce v15 feature stack on 2024 fold.
  - Add 12 snapshot-difference features:
      ly_{success,reverse,ball,middle}_rel
      cur_minus_ly_{success,reverse,ball,middle}
      ly_minus_old_{success,reverse,ball,middle}
  - Compare 91 vs 103 for HGB, CatBoost, LGBM classifier, LGBM regressor, XGBoost.

This is a fold experiment only, not a submit artifact builder.
"""

import os
import sys
import time
import warnings

sys.stdout.reconfigure(encoding="utf-8")
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import xgboost as xgb
import joblib

from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon


SEED = 42
DATA_PATH = "../data/train.csv"
OUT_DIR = "phase42_preds"
FEATURE_CACHE = os.path.join(OUT_DIR, "phase42_feature_cache.pkl")
SUMMARY_PATH = os.path.join(OUT_DIR, "phase42_time103_model_summary.csv")
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
RATE_KEYS = ["success", "reverse", "ball", "middle"]


def score(y, p):
    return max(0.0, evaluate(y, np.clip(p, 0.0, 1.0))["bss"] * 1e5)


def league_season_rates(ly_table, seasons):
    rows = []
    for s in seasons:
        cur = ly_table[ly_table["season"] == s]
        if cur.empty:
            continue
        n = cur["N_end"].to_numpy(np.float64)
        rows.append({
            "season": s,
            "success": float(cur["S_end"].sum() / max(cur["N_end"].sum(), 1.0)),
            "reverse": float(cur["R_end"].sum() / max(cur["N_end"].sum(), 1.0)),
            "ball": float(cur["B_end"].sum() / max(cur["N_end"].sum(), 1.0)),
            "middle": float(cur["M_end"].sum() / max(cur["N_end"].sum(), 1.0)),
        })
    return pd.DataFrame(rows).set_index("season")


def _lookup_cum(df, pivots, season_offset):
    pid = df["pitcher_id"].to_numpy()
    season = df["season"].to_numpy() + season_offset
    idx = pd.MultiIndex.from_arrays([pid, season])
    out = {}
    for c in ["N_end", "S_end", "R_end", "B_end", "M_end"]:
        out[c] = np.nan_to_num(pivots[c].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
    return out


def _rates_from_counts(num, den, fallback):
    raw = np.divide(num, den, out=np.full_like(den, np.nan, dtype=np.float64), where=den > 0)
    return np.nan_to_num(raw, nan=fallback)


def build_time103_features(df, ly_table, global_rates, seasons_range, k_cur=15.0, k_ly=30.0):
    """Build 12 row-safe time-snapshot features."""
    pivots = {c: ly_table.pivot(index="pitcher_id", columns="season", values=c)
                          .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
              for c in ["N_end", "S_end", "R_end", "B_end", "M_end"]}
    lg = league_season_rates(ly_table, seasons_range)

    c0 = _lookup_cum(df, pivots, -0)   # latest completed season available in asof may exceed this; only for old.
    c1 = _lookup_cum(df, pivots, -1)   # end of season-1
    c2 = _lookup_cum(df, pivots, -2)   # end of season-2
    c3 = _lookup_cum(df, pivots, -3)   # end of season-3

    # Current cumulative counts from row-local asof values.
    n_now = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    now = {
        "N_end": n_now,
        "S_end": np.round(df["asof_pitcher_success_rate"].fillna(0).to_numpy(np.float64) * n_now),
        "R_end": np.round(df["asof_pitcher_reverse_rate"].fillna(0).to_numpy(np.float64) * n_now),
        "B_end": np.round(df["asof_pitcher_ball_rate"].fillna(0).to_numpy(np.float64) * n_now),
        "M_end": np.round(df["asof_pitcher_middle_rate"].fillna(0).to_numpy(np.float64) * n_now),
    }

    n_cur = np.clip(now["N_end"] - c1["N_end"], 0, None)
    n_ly = np.clip(c1["N_end"] - c2["N_end"], 0, None)
    n_old = c2["N_end"]

    out = pd.DataFrame(index=df.index)
    prev_season = df["season"].to_numpy() - 1

    for key, col in [("success", "S_end"), ("reverse", "R_end"), ("ball", "B_end"), ("middle", "M_end")]:
        gm = float(global_rates[key])
        cnt_cur = np.clip(now[col] - c1[col], 0, None)
        cnt_ly = np.clip(c1[col] - c2[col], 0, None)
        cnt_old = np.clip(c2[col], 0, None)

        cur_raw = _rates_from_counts(cnt_cur, n_cur, gm)
        ly_raw = _rates_from_counts(cnt_ly, n_ly, gm)
        old_raw = _rates_from_counts(cnt_old, n_old, gm)

        cur_sm = (n_cur * cur_raw + k_cur * gm) / (n_cur + k_cur)
        ly_sm = (n_ly * ly_raw + k_ly * gm) / (n_ly + k_ly)
        old_sm = (n_old * old_raw + k_ly * gm) / (n_old + k_ly)

        rel = []
        for ps in prev_season:
            if ps in lg.index:
                rel.append(float(lg.loc[ps, key]))
            else:
                rel.append(gm)
        rel = np.asarray(rel, dtype=np.float64)

        out[f"ly_{key}_rel"] = ly_sm - rel
        out[f"cur_minus_ly_{key}"] = cur_sm - ly_sm
        out[f"ly_minus_old_{key}"] = ly_sm - old_sm

    return out.astype(np.float64)


def build_v15_fold_features(df):
    if os.path.exists(FEATURE_CACHE):
        print(f"Load feature cache: {FEATURE_CACHE}", flush=True)
        return joblib.load(FEATURE_CACHE)

    t0 = time.time()
    df = df.copy()
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    g = float(df[TARGET_COL].mean())
    seasons_range = sorted(df["season"].unique().tolist())
    tr_idx = df[df["season"] <= 2023].index
    va_idx = df[df["season"] == 2024].index
    tr_df = df.loc[tr_idx].reset_index(drop=True)
    va_df = df.loc[va_idx].reset_index(drop=True)

    print(f"Build v15 fold features: train={len(tr_df):,} valid={len(va_df):,}", flush=True)
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(tr_df)
    Xb_tr = fb.transform_train_oof(tr_df).reset_index(drop=True)
    Xb_va = fb.transform(va_df).reset_index(drop=True)

    se = build_season_end_table(df)
    dins = transform_inseason(df, se, g, seasons_range)
    piv = _pivots_from_table(se, seasons_range)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)

    dplt = transform_platoon(df, build_platoon_table(df), prior, seasons_range, k=K_PLATOON)
    dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), prior,
                            seasons_range, k=K_INNING)
    dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), seasons_range),
                              prior, g, seasons_range)
    ly_table = build_lastyear_table(df)
    gr = build_global_rates(df)
    dly = transform_lastyear(df, ly_table, gr, seasons_range, k=30.0)
    d103 = build_time103_features(df, ly_table, gr, seasons_range)

    def stack(base, idx_):
        idx_list = list(idx_)
        X = pd.concat([
            base.reset_index(drop=True),
            dins.loc[idx_list, INS].reset_index(drop=True),
            dplt.loc[idx_list].reset_index(drop=True),
            dinn.loc[idx_list].reset_index(drop=True),
            dpt.loc[idx_list].reset_index(drop=True),
        ], axis=1).astype(np.float64)
        X = pd.concat([X, add_crosses(X), dly.loc[idx_list].reset_index(drop=True)], axis=1)
        X103 = pd.concat([X, d103.loc[idx_list].reset_index(drop=True)], axis=1)
        return X.astype(np.float64), X103.astype(np.float64)

    X91_tr, X103_tr = stack(Xb_tr, tr_idx)
    X91_va, X103_va = stack(Xb_va, va_idx)
    print(f"  features: {X91_tr.shape[1]} -> {X103_tr.shape[1]} ({time.time()-t0:.0f}s)", flush=True)
    print("  added:", ", ".join(X103_tr.columns[-12:]), flush=True)
    payload = (X91_tr, X91_va, X103_tr, X103_va,
               tr_df[TARGET_COL].to_numpy(), va_df[TARGET_COL].to_numpy())
    os.makedirs(OUT_DIR, exist_ok=True)
    joblib.dump(payload, FEATURE_CACHE)
    print(f"  cached: {FEATURE_CACHE}", flush=True)
    return payload


def fit_hgb(Xtr, ytr, Xva, kind):
    grids = [
        dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0),
        dict(max_depth=5, max_leaf_nodes=31, max_iter=650, learning_rate=0.025, l2_regularization=2.0),
    ]
    best = None
    for i, p in enumerate(grids):
        t = time.time()
        m = HistGradientBoostingClassifier(**p, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=20, random_state=SEED).fit(Xtr, ytr)
        pred = m.predict_proba(Xva)[:, 1]
        s = score(yva_global, pred)
        print(f"    HGB {kind} cfg{i} score={s:.1f} ({time.time()-t:.0f}s)", flush=True)
        if best is None or s > best[0]:
            best = (s, pred, f"cfg{i}", m)
    return best


def fit_cat(Xtr, ytr, Xva, kind):
    tr_i, es_i = time_split_es(len(Xtr))
    grids = [
        dict(depth=6, l2_leaf_reg=5.0, learning_rate=0.03),
        dict(depth=6, l2_leaf_reg=15.0, learning_rate=0.03),
    ]
    best = None
    for i, p in enumerate(grids):
        t = time.time()
        m = CatBoostClassifier(iterations=3000, random_seed=SEED, verbose=0,
                               early_stopping_rounds=50, min_data_in_leaf=200,
                               loss_function="Logloss", **p)
        m.fit(Xtr.iloc[tr_i], ytr[tr_i], eval_set=(Xtr.iloc[es_i], ytr[es_i]))
        pred = m.predict_proba(Xva)[:, 1]
        s = score(yva_global, pred)
        print(f"    Cat {kind} cfg{i} score={s:.1f} best_iter={m.best_iteration_} ({time.time()-t:.0f}s)", flush=True)
        if best is None or s > best[0]:
            best = (s, pred, f"cfg{i}", m)
    return best


def fit_lgbm_cls(Xtr, ytr, Xva, kind):
    tr_i, es_i = time_split_es(len(Xtr))
    grids = [
        dict(num_leaves=31, max_depth=6, learning_rate=0.02, min_child_samples=80,
             reg_alpha=0.05, reg_lambda=1.0, colsample_bytree=0.8, subsample=0.9),
        dict(num_leaves=64, max_depth=12, learning_rate=0.005571638320335239,
             min_child_samples=28, subsample=0.9017762093981382,
             colsample_bytree=0.5291780969405919, reg_alpha=0.07089938907781941,
             reg_lambda=0.009306216375166584, min_split_gain=0.4888649495163153, max_bin=127),
    ]
    best = None
    for i, p in enumerate(grids):
        t = time.time()
        m = LGBMClassifier(n_estimators=3000, random_state=SEED, n_jobs=-1, verbosity=-1,
                           subsample_freq=1, **p)
        m.fit(Xtr.iloc[tr_i], ytr[tr_i], eval_set=[(Xtr.iloc[es_i], ytr[es_i])],
              eval_metric="binary_logloss",
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        pred = m.predict_proba(Xva)[:, 1]
        s = score(yva_global, pred)
        print(f"    LGBMcls {kind} cfg{i} score={s:.1f} best_iter={m.best_iteration_} ({time.time()-t:.0f}s)", flush=True)
        if best is None or s > best[0]:
            best = (s, pred, f"cfg{i}", m)
    return best


def fit_lgbm_reg(Xtr, ytr, Xva, kind):
    tr_i, es_i = time_split_es(len(Xtr))
    grids = [
        dict(num_leaves=64, max_depth=12, learning_rate=0.005571638320335239,
             min_child_samples=28, subsample=0.9017762093981382,
             colsample_bytree=0.5291780969405919, reg_alpha=0.07089938907781941,
             reg_lambda=0.009306216375166584, min_split_gain=0.4888649495163153, max_bin=127),
        dict(num_leaves=31, max_depth=6, learning_rate=0.015, min_child_samples=80,
             reg_alpha=0.05, reg_lambda=1.0, colsample_bytree=0.8, subsample=0.9),
    ]
    best = None
    for i, p in enumerate(grids):
        t = time.time()
        m = LGBMRegressor(n_estimators=3000, random_state=SEED, n_jobs=-1, verbosity=-1,
                          subsample_freq=1, **p)
        m.fit(Xtr.iloc[tr_i], ytr[tr_i].astype(np.float64),
              eval_set=[(Xtr.iloc[es_i], ytr[es_i].astype(np.float64))], eval_metric="l2",
              callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        pred = np.clip(m.predict(Xva), 0.0, 1.0)
        s = score(yva_global, pred)
        print(f"    LGBMreg {kind} cfg{i} score={s:.1f} best_iter={m.best_iteration_} ({time.time()-t:.0f}s)", flush=True)
        if best is None or s > best[0]:
            best = (s, pred, f"cfg{i}", m)
    return best


def fit_xgb(Xtr, ytr, Xva, kind):
    tr_i, es_i = time_split_es(len(Xtr))
    grids = [
        dict(max_depth=5, learning_rate=0.02, min_child_weight=40, subsample=0.9,
             colsample_bytree=0.8, reg_alpha=0.05, reg_lambda=1.0),
        dict(max_depth=4, learning_rate=0.025, min_child_weight=80, subsample=0.85,
             colsample_bytree=0.9, reg_alpha=0.0, reg_lambda=2.0),
    ]
    best = None
    for i, p in enumerate(grids):
        t = time.time()
        m = xgb.XGBClassifier(objective="binary:logistic", n_estimators=2500,
                              tree_method="hist", max_bin=256, random_state=SEED,
                              n_jobs=-1, early_stopping_rounds=100, eval_metric="logloss", **p)
        m.fit(Xtr.iloc[tr_i], ytr[tr_i], eval_set=[(Xtr.iloc[es_i], ytr[es_i])], verbose=False)
        pred = m.predict_proba(Xva)[:, 1]
        s = score(yva_global, pred)
        print(f"    XGB {kind} cfg{i} score={s:.1f} best_iter={m.best_iteration} ({time.time()-t:.0f}s)", flush=True)
        if best is None or s > best[0]:
            best = (s, pred, f"cfg{i}", m)
    return best


def run_model_family(name, fn, X91_tr, X91_va, X103_tr, X103_va, ytr):
    print(f"\n[{name}]", flush=True)
    b91 = fn(X91_tr, ytr, X91_va, "91")
    b103 = fn(X103_tr, ytr, X103_va, "103")
    return {
        "model": name,
        "score_91": b91[0],
        "score_103": b103[0],
        "delta": b103[0] - b91[0],
        "best_91": b91[2],
        "best_103": b103[2],
        "pred_91": b91[1],
        "pred_103": b103[1],
    }


def save_partial(rows):
    out_rows = [{k: v for k, v in r.items() if not k.startswith("pred_")} for r in rows]
    pd.DataFrame(out_rows).to_csv(SUMMARY_PATH, index=False, encoding="utf-8")


if __name__ == "__main__":
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    X91_tr, X91_va, X103_tr, X103_va, ytr, yva = build_v15_fold_features(df)
    yva_global = yva

    results = []
    for name, fn in [
        ("HGB", fit_hgb),
        ("CatBoost", fit_cat),
        ("LGBMClassifier", fit_lgbm_cls),
        ("LGBMRegressor", fit_lgbm_reg),
        ("XGBoost", fit_xgb),
    ]:
        res = run_model_family(name, fn, X91_tr, X91_va, X103_tr, X103_va, ytr)
        results.append(res)
        save_partial(results)
        print(f"  => {name}: {res['score_91']:.1f} -> {res['score_103']:.1f} "
              f"delta={res['delta']:+.1f}", flush=True)

    # v15-style 50:50 blend: use the tuned HGB and CatBoost predictions from above.
    by_name = {r["model"]: r for r in results}
    blend91 = 0.5 * by_name["HGB"]["pred_91"] + 0.5 * by_name["CatBoost"]["pred_91"]
    blend103 = 0.5 * by_name["HGB"]["pred_103"] + 0.5 * by_name["CatBoost"]["pred_103"]
    blend_row = {
        "model": "v15_blend_HGB_Cat_50_50",
        "score_91": score(yva, blend91),
        "score_103": score(yva, blend103),
        "delta": score(yva, blend103) - score(yva, blend91),
        "best_91": f"{by_name['HGB']['best_91']}+{by_name['CatBoost']['best_91']}",
        "best_103": f"{by_name['HGB']['best_103']}+{by_name['CatBoost']['best_103']}",
    }
    results.append(blend_row)
    save_partial(results)

    out_rows = [{k: v for k, v in r.items() if not k.startswith("pred_")} for r in results]
    summary = pd.DataFrame(out_rows)
    summary.to_csv(SUMMARY_PATH, index=False, encoding="utf-8")
    print("\nSUMMARY", flush=True)
    print(summary.to_string(index=False), flush=True)
    print(f"\nSaved: {OUT_DIR}/phase42_time103_model_summary.csv", flush=True)
    print(f"Total {time.time()-t0:.0f}s", flush=True)
