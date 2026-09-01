"""SHAP dependence 분석: v18/v20/v22의 신규 feature + 공통 초반 feature(inning/platoon/inseason)가
season/game_type 하위 구간에서 부호가 뒤집히는지(=feature masking/bad interaction) 점검.

각 버전의 train_final_vXX.py와 동일한 파이프라인으로 X를 재구성하고, 저장된 CatBoost 모델에
Pool(X,y)를 넣어 get_feature_importance(type='ShapValues')로 SHAP을 뽑는다(외부 shap 패키지 불필요).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

CACHE_DIR = "shap_cache"

from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from inning_split import (K_INNING, build_inning_offset, build_inning_table, transform_inning)
from inseason import (build_season_end_table, transform_inseason, _pivots_from_table)
from lastyear import (build_global_rates, build_lastyear_table, transform_lastyear)
from pitchtype import (build_matched, build_pitchtype_tables, transform_pitchtype)
from platoon import build_platoon_table, transform_platoon, K_PLATOON

DATA_PATH = "../data/train.csv"
MODEL_DIR = "../submit/model"


def build_common(df):
    g = float(df[TARGET_COL].mean())
    sr = sorted(df["season"].unique().tolist())

    fb = FeatureBuilder(seed=42, include_raw_rates=False).fit(df)
    X_base = fb.transform_train_oof(df).reset_index(drop=True)

    se = build_season_end_table(df)
    X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)

    pt = build_platoon_table(df)
    X_plt = transform_platoon(df, pt, prior, sr, k=K_PLATOON).reset_index(drop=True)
    it, io = build_inning_table(df), build_inning_offset(df)
    X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)

    matched = build_matched(df)
    pt_tables = build_pitchtype_tables(matched, sr)
    X_pt = transform_pitchtype(df, pt_tables, prior, g, sr).reset_index(drop=True)

    gr = build_global_rates(df)
    ly_tbl = build_lastyear_table(df)
    X_ly = transform_lastyear(df, ly_tbl, gr, sr, k=30.0).reset_index(drop=True)

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C, X_ly], axis=1)
    return X, sr


def build_X_v18(df):
    from train_final_v18 import hidden_denominator_features
    X, sr = build_common(df)
    X_den = hidden_denominator_features(df).reset_index(drop=True)
    X = pd.concat([X, X_den], axis=1)
    new_feats = list(X_den.columns)
    return X, new_feats


def build_X_v20(df):
    from career_volatility import K_VOL, build_volatility_table, transform_volatility
    X, sr = build_common(df)
    se = build_season_end_table(df)
    vol_tbl = build_volatility_table(se)
    X_vol = transform_volatility(df, vol_tbl, sr, k=K_VOL).reset_index(drop=True)
    X = pd.concat([X, X_vol], axis=1)
    new_feats = list(X_vol.columns)
    return X, new_feats


def build_X_v22(df):
    from arsenal_entropy import K_ARSENAL, transform_arsenal
    X, sr = build_common(df)
    arsenal_global_mix = {c: float(df[c].mean(skipna=True)) for c in
                          ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]}
    X_ars = transform_arsenal(df, global_mix=arsenal_global_mix, k=K_ARSENAL).reset_index(drop=True)
    X = pd.concat([X, X_ars], axis=1)
    new_feats = list(X_ars.columns)
    return X, new_feats


BUILDERS = {"v18": build_X_v18, "v20": build_X_v20, "v22": build_X_v22}
COMMON_OLD_FEATS = [
    "inseason_success_smooth", "inseason_reverse_smooth", "inseason_n",
    "platoon_diff", "platoon_n", "inning_diff", "inning_n",
    "pt_pred", "pt_dev", "pt_n",
    "ly_success", "ly_reverse", "ly_ball", "ly_middle", "ly_n", "ly_minus_career",
    "ability_composite",
    "pitcher_team_id_te", "batter_team_id_te",
    "cat_game_type", "season", "asof_pitcher_success_rate_smooth",
    "x_ability_here", "x_kal_minus_career", "x_prev5_minus_career",
    "x_rev_over_succ", "x_p_over_b", "x_platoon_x_samehand",
]


def get_cat_models(artifacts):
    if "cats" in artifacts:
        return artifacts["cats"]
    return [artifacts["cat"]]


def shap_matrix(models, X, y, feature_order):
    Xo = X[feature_order]
    pool = Pool(Xo, y)
    mats = []
    for m in models:
        sv = m.get_feature_importance(pool, type="ShapValues")
        mats.append(sv[:, :-1])  # drop bias column
    return np.mean(mats, axis=0), feature_order


def report_feature(name, X, shap_vals, feat_idx, df):
    s = shap_vals[:, feat_idx]
    v = X[name].to_numpy()
    corr_all = np.corrcoef(v, s)[0, 1] if np.std(v) > 0 else float("nan")
    print(f"\n--- {name} ---  overall mean|SHAP|={np.abs(s).mean():.5f}  corr(value,shap)={corr_all:+.3f}")

    print("  by season:")
    for season, idx in df.groupby("season").groups.items():
        ii = df.index.get_indexer(idx)
        vv, ss = v[ii], s[ii]
        c = np.corrcoef(vv, ss)[0, 1] if np.std(vv) > 0 else float("nan")
        print(f"    season={season}: mean_shap={ss.mean():+.5f}  corr={c:+.3f}  n={len(ii)}")

    print("  by game_type:")
    for gt, idx in df.groupby("game_type").groups.items():
        ii = df.index.get_indexer(idx)
        vv, ss = v[ii], s[ii]
        c = np.corrcoef(vv, ss)[0, 1] if np.std(vv) > 0 else float("nan")
        print(f"    game_type={gt}: mean_shap={ss.mean():+.5f}  corr={c:+.3f}  n={len(ii)}")


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    y = df[TARGET_COL].to_numpy()

    os.makedirs(CACHE_DIR, exist_ok=True)
    for tag, builder in BUILDERS.items():
        print(f"\n{'='*70}\n{tag}\n{'='*70}", flush=True)
        cache_path = f"{CACHE_DIR}/{tag}.joblib"
        artifacts = joblib.load(f"{MODEL_DIR}/model_artifacts_{tag}.pkl")
        feature_order = artifacts["feature_order"]

        if os.path.exists(cache_path):
            print(f"cache hit -> {cache_path}", flush=True)
            cached = joblib.load(cache_path)
            X, shap_vals, order = cached["X"], cached["shap_vals"], cached["order"]
            new_feats = cached["new_feats"]
        else:
            models = get_cat_models(artifacts)
            X, new_feats = builder(df)
            print(f"X built ({X.shape[1]} cols, {time.time()-t0:.0f}s), computing SHAP for {len(models)} model(s)...", flush=True)
            shap_vals, order = shap_matrix(models, X, y, feature_order)
            joblib.dump({"X": X, "shap_vals": shap_vals, "order": order, "new_feats": new_feats}, cache_path)
            print(f"cached -> {cache_path} ({time.time()-t0:.0f}s)", flush=True)

        idx_map = {c: i for i, c in enumerate(order)}

        targets = new_feats + COMMON_OLD_FEATS
        for name in targets:
            if name not in idx_map:
                continue
            report_feature(name, X, shap_vals, idx_map[name], df)

    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
