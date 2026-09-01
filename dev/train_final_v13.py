"""v13 = v12(963.796 실증) + 작년피처 + K 재설정 + CatBoost 3시드 + 20:80 비중.

채택 근거 (폴드 노이즈 SD~11점이 확인되어, 폴드 단독 근거는 전부 기각):
  1) in-season K 15 -> 60
     이론 최적 87 / 전반->후반 검증 최적 150 (둘 다 폴드와 무관한 독립 측정)
  2) platoon K 520 -> 2500
     3시드평균 823.3(K=1200) -> 835.3(K=2500). in-season과 같은 방향("더 강하게 축소")
  3) 작년 한 시즌 피처 7개
     다음시즌 예측 R^2: 커리어만 0.2821 / 작년만 0.2841 / 둘다 0.3051 (독립 측정)
     reverse_rate 상관 -0.4939로 success(+0.5311)에 맞먹는데 미활용이었음
  4) CatBoost 3시드 평균 — 편향 불변, 분산만 감소(이론적 보장)
  5) HGB 0.2 : Cat 0.8 — K=15/60/150/300 네 조건 전부에서 50:50보다 우세

기각한 것: inning K 변경(비단조=노이즈), cat depth 변경(6이 최적 확인),
           시대보정(R^2 +2.7%, 변수간 상관 0.969)
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import inseason as INS_MOD
from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from inning_split import (K_INNING, build_inning_offset, build_inning_table, transform_inning,
                          export_stats as inning_export_stats)
from inseason import build_season_end_table, _pivots_from_table, export_stats as inseason_export_stats
from lastyear import (build_global_rates, build_lastyear_table, transform_lastyear,
                      export_stats as lastyear_export_stats)
from phase2_common import time_split_es
from platoon import build_platoon_table, transform_platoon, export_stats as platoon_export_stats

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42
K_INSEASON = 60.0
K_PLATOON_NEW = 2500.0
K_LASTYEAR = 30.0
CAT_SEEDS = (42, 7, 2024)
W_HGB, W_CAT = 0.2, 0.8

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

    print(f"\n[2] in-season (K={K_INSEASON:.0f}, 기존 15에서 상향)...", flush=True)
    INS_MOD.K_SMOOTH = K_INSEASON
    se = build_season_end_table(df)
    X_ins = INS_MOD.transform_inseason(df, se, g, sr).reset_index(drop=True)
    piv = _pivots_from_table(se, sr)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)

    print(f"\n[3] platoon (K={K_PLATOON_NEW:.0f}, 기존 520에서 상향) + inning...", flush=True)
    pt = build_platoon_table(df)
    X_plt = transform_platoon(df, pt, prior, sr, k=K_PLATOON_NEW).reset_index(drop=True)
    it, io = build_inning_table(df), build_inning_offset(df)
    X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)

    print("\n[4] 작년 한 시즌 피처 (신규 7개)...", flush=True)
    gr = build_global_rates(df)
    lyt = build_lastyear_table(df)
    X_ly = transform_lastyear(df, lyt, gr, sr, k=K_LASTYEAR).reset_index(drop=True)

    X = pd.concat([X_base, X_ins, X_plt, X_inn, X_ly], axis=1).astype(np.float64)
    C = add_crosses(X)
    X = pd.concat([X, C], axis=1)
    print(f"\n교차항 {C.shape[1]}개 -> 최종 피처 수={X.shape[1]}  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[5] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(
        max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED).fit(X, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n[6] CatBoost {len(CAT_SEEDS)}시드 학습...", flush=True)
    tr_i, es_i = time_split_es(len(X))
    cats = []
    for s in CAT_SEEDS:
        cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=15.0,
                                random_seed=s, verbose=0, early_stopping_rounds=50,
                                min_data_in_leaf=200, loss_function="Logloss")
        cb.fit(X.iloc[tr_i], y[tr_i], eval_set=(X.iloc[es_i], y[es_i]))
        cats.append(cb)
        print(f"  seed={s} best_iter={cb.best_iteration_} ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    artifacts = {
        "hgb": hgb, "cats": cats, "w_hgb": W_HGB, "w_cat": W_CAT,
        "stats": fb.export_stats(),
        "inseason_stats": dict(inseason_export_stats(se, g, sr), k_smooth=K_INSEASON),
        "platoon_stats": platoon_export_stats(pt, sr, k=K_PLATOON_NEW),
        "inning_stats": inning_export_stats(it, io, sr, k=K_INNING),
        "lastyear_stats": lastyear_export_stats(lyt, gr, sr, k=K_LASTYEAR),
        "feature_order": list(X.columns),
    }
    out = os.path.join(OUT_DIR, "model_artifacts_v13.pkl")
    joblib.dump(artifacts, out)
    print(f"\n저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)  총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
