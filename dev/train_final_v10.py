"""v10 = v7c 피처셋(batter_asof 유지) + 칼만 교체 + HGB/CatBoost 앙상블 = 66피처.

실제 리더보드 3점으로 분해한 결과에 근거:
  v7c                          948.970
  v9  = v7c - batter_asof      922.415  -> 제거 효과 = -26.6 (로컬 ablation은 +33.0이었음)
  v8  = v9 + 칼만 + CatBoost     939.875  -> 칼만+CatBoost = +17.5
따라서 batter_asof는 유지하고 칼만+CatBoost만 얹는다.

교훈: 단일 폴드 로컬 검증은 피처셋 변경에 대해 부호조차 못 맞춘다(부트스트랩 CI가 좁아도).
      2024 폴드 vs 2025 test의 분포 차이는 부트스트랩이 못 잡는 계통 오차다.

--- 이하 v8 원본 설명 ---
v8 = F2 피처셋(62) + HGB + CatBoost 앙상블.

phase23 2024폴드: 841.1 (기준 v7c HGB 751.3 대비 +89.8) -> 예상 실제 ~991점

F2 구성 = 58 base - batter_asof(4) + 칼만(4) + platoon(2) + inning(2) = 62피처
  - batter_asof 제거: ablation +33.0 (타자 정보는 이 타깃에 해로움)
  - inseason 5개 -> 칼만 4개 교체: +10.7 (추가하면 -8.8, 교체해야 이득)
  - CatBoost: 단독으로 HGB보다 훨씬 강함 (836.6 vs 794.9)
  - LGBM 제외: hgb+cat(841.1) > hgb+lgbm+cat(827.2)
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from features import FeatureBuilder, TARGET_COL
from inning_split import (K_INNING, build_inning_offset, build_inning_table, transform_inning,
                          export_stats as inning_export_stats)
from inseason import (K_SMOOTH, build_season_end_table, transform_inseason, _pivots_from_table,
                      export_stats as inseason_export_stats)
from kalman_ability import (build_kalman_table, estimate_process_noise, transform_kalman,
                            export_stats as kalman_export_stats)
from phase2_common import time_split_es
from platoon import (K_PLATOON, build_platoon_table, transform_platoon,
                     export_stats as platoon_export_stats)

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42
W_HGB, W_CAT = 0.5, 0.5
BATTER_ASOF = ["flag_asof_batter_n_zero", "asof_batter_n",
               "asof_batter_success_rate_smooth", "asof_batter_middle_rate_smooth"]

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

    print("\n[2] in-season 내부값 + prior (칼만/플래툰/이닝 재료)...", flush=True)
    se = build_season_end_table(df)
    dins = transform_inseason(df, se, g, sr)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)

    print("\n[3] 칼만 (in-season 5개를 대체)...", flush=True)
    q = estimate_process_noise(df)
    th, P = build_kalman_table(df, sr, q, g)
    n_season = np.expm1(dins["inseason_n"].to_numpy(np.float64))
    sm = dins["inseason_success_smooth"].to_numpy(np.float64)
    raw = np.clip(np.where(n_season > 0,
                           (sm * (n_season + K_SMOOTH) - K_SMOOTH * prior) / np.maximum(n_season, 1e-9),
                           np.nan), 0, 1)
    X_kal = transform_kalman(df, th, P, g, inseason_n=n_season, inseason_rate=raw).reset_index(drop=True)
    print(f"  q={q:.6f}  {X_kal.shape[1]}피처 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4] platoon + inning...", flush=True)
    pt = build_platoon_table(df)
    X_plt = transform_platoon(df, pt, prior, sr, k=K_PLATOON).reset_index(drop=True)
    it, io = build_inning_table(df), build_inning_offset(df)
    X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)

    X = pd.concat([X_base, X_kal, X_plt, X_inn], axis=1).astype(np.float64)
    print(f"\n최종 피처 수={X.shape[1]}  (batter_asof 유지)", flush=True)

    print("\n[5] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(
        max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED).fit(X, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[6] CatBoost 학습...", flush=True)
    tr_i, es_i = time_split_es(len(X))
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            random_seed=SEED, verbose=0, early_stopping_rounds=50,
                            min_data_in_leaf=200, loss_function="Logloss")
    cb.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
    print(f"  완료 best_iter={cb.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    artifacts = {
        "hgb": hgb, "cat": cb, "w_hgb": W_HGB, "w_cat": W_CAT,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(se, g, sr),
        "kalman_stats": kalman_export_stats(th, P, g, q, sr),
        "platoon_stats": platoon_export_stats(pt, sr, k=K_PLATOON),
        "inning_stats": inning_export_stats(it, io, sr, k=K_INNING),
        "drop_cols": [],
        "feature_order": list(X.columns),
        "k_smooth_inseason": K_SMOOTH,
    }
    out = os.path.join(OUT_DIR, "model_artifacts_v10.pkl")
    joblib.dump(artifacts, out)
    print(f"\n저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)  총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
