"""v16 = v15 + time-opened 12 features = 103 features.

Local phase42 result was negative on 2024 fold, but this builds a separate
submission artifact for leaderboard probing as requested.
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from inning_split import (K_INNING, build_inning_offset, build_inning_table, transform_inning,
                          export_stats as inning_export_stats)
from inseason import (build_season_end_table, transform_inseason, _pivots_from_table,
                      export_stats as inseason_export_stats)
from lastyear import (build_global_rates, build_lastyear_table, transform_lastyear,
                      export_stats as lastyear_export_stats)
from phase2_common import time_split_es
from pitchtype import (K_CONTROL, K_MIX, build_matched, build_pitchtype_tables, transform_pitchtype,
                       export_stats as pitchtype_export_stats)
from platoon import build_platoon_table, transform_platoon, export_stats as platoon_export_stats, K_PLATOON
from time103 import transform_time103, export_stats as time103_export_stats

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42

HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                  early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)

_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")


def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG_TYPES:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strip_rng(v, seen, depth + 1)


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    y = df[TARGET_COL].to_numpy()
    g = float(df[TARGET_COL].mean())
    sr = sorted(df["season"].unique().tolist())
    print(f"train={len(df):,}  season {sr[0]}~{sr[-1]}", flush=True)

    print("\n[1] base 피처...", flush=True)
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(df)
    X_base = fb.transform_train_oof(df).reset_index(drop=True)
    print(f"  {X_base.shape[1]}피처 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[2] in-season (v12/v15와 동일 K=15)...", flush=True)
    se = build_season_end_table(df)
    X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)

    print("\n[3] platoon(K=520) + inning(K=570) - v15와 동일...", flush=True)
    pt = build_platoon_table(df)
    X_plt = transform_platoon(df, pt, prior, sr, k=K_PLATOON).reset_index(drop=True)
    it, io = build_inning_table(df), build_inning_offset(df)
    X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)

    print("\n[4] 구종 피처 3개...", flush=True)
    matched = build_matched(df)
    pt_tables = build_pitchtype_tables(matched, sr)
    X_pt = transform_pitchtype(df, pt_tables, prior, g, sr).reset_index(drop=True)
    print(f"  매칭 {len(matched):,}행  pt_dev SD={X_pt.pt_dev.std():.5f}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[5] 작년 7개 + time103 12개...", flush=True)
    gr = build_global_rates(df)
    ly_tbl = build_lastyear_table(df)
    X_ly = transform_lastyear(df, ly_tbl, gr, sr, k=30.0).reset_index(drop=True)
    X_time = transform_time103(df, ly_tbl, gr, sr).reset_index(drop=True)
    print(f"  ly_reverse SD={X_ly['ly_reverse'].std():.5f}  time103={X_time.shape[1]} ({time.time()-t0:.0f}s)", flush=True)

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C, X_ly, X_time], axis=1)
    print(f"\n교차항 {C.shape[1]}개 + 작년 {X_ly.shape[1]}개 + time103 {X_time.shape[1]}개 -> 최종 {X.shape[1]}피처", flush=True)

    print("\n[6] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[7] CatBoost 학습 (단일 시드, v15와 동일)...", flush=True)
    tr_i, es_i = time_split_es(len(X))
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            random_seed=SEED, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
    print(f"  완료 best_iter={cb.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(hgb)
    os.makedirs(OUT_DIR, exist_ok=True)
    artifacts = {
        "hgb": hgb,
        "cat": cb,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(se, g, sr),
        "platoon_stats": platoon_export_stats(pt, sr, k=K_PLATOON),
        "inning_stats": inning_export_stats(it, io, sr, k=K_INNING),
        "pitchtype_stats": pitchtype_export_stats(pt_tables, g, sr, k_control=K_CONTROL, k_mix=K_MIX),
        "lastyear_stats": lastyear_export_stats(ly_tbl, gr, sr, k=30.0),
        "time103_stats": time103_export_stats(ly_tbl, gr, sr),
        "feature_order": list(X.columns),
        "w_hgb": 0.5,
        "w_cat": 0.5,
    }
    out = os.path.join(OUT_DIR, "model_artifacts_v16.pkl")
    joblib.dump(artifacts, out)
    print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB) w_hgb=0.5 w_cat=0.5", flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
