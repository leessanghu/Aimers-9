"""v7 = v4(925.908 실증) + platoon(K=520) 2개 피처 = 65피처.

phase9_platoon.py 3폴드 검증(rf015_hgb085): 2022 +18.1 / 2023 -3.6 / 2024 +35.2
hgb 단독: 2022 +14.3 / 2023 +8.2 / 2024 +38.0 (3폴드 전부 양수, rf보다 안정적)

근거: 플래툰 스플릿(투수x타자손) 개인차 진짜SD=0.0438 (투수 실력 개인차 0.0555의 79%),
주최측 asof_* 컬럼엔 없는 조건부 정보 -> in-season과 같은 '진짜 새 정보' 범주.

leakage: platoon_table은 (pitcher, batter_hand, season) 누적을 train 라벨로만 만들고,
각 행은 자기 투수의 '직전 시즌 끝'까지 누적만 조회한다. 같은 시즌 다른 행 참조 없음.

동일 학습으로 배합 비중만 다른 2개 후보 저장 (재학습 1회로 재확인 없이 A/B 제출 가능):
  v7a: w_rf=0.15 w_hgb=0.85 (v4와 동일 비중)
  v7b: w_rf=0.00 w_hgb=1.00 (hgb 단독 - platoon에서 더 안정적이었던 쪽)
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from inseason import build_season_end_table, transform_inseason, _pivots_from_table, export_stats as inseason_export_stats
from platoon import build_platoon_table, transform_platoon, export_stats as platoon_export_stats, K_PLATOON

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
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


def get_prior_rate(df, season_end_table, global_success_rate, seasons_range):
    pivots = _pivots_from_table(season_end_table, seasons_range)
    lookup_idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    vals = pivots["rate"].reindex(lookup_idx).to_numpy()
    return pd.Series(vals).fillna(global_success_rate).to_numpy(np.float64)


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    print(f"전체 train={len(df):,}  season {df['season'].min()}~{df['season'].max()}", flush=True)
    y = df[TARGET_COL].to_numpy()

    print("\n[1] FeatureBuilder (58피처 + team_te)...", flush=True)
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(df)
    X_base = fb.transform_train_oof(df).reset_index(drop=True)
    print(f"  {X_base.shape[1]}피처  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[2] in-season 피처 (5개)...", flush=True)
    season_end_table = build_season_end_table(df)
    seasons_range = sorted(df["season"].unique().tolist())
    global_success_rate = float(df[TARGET_COL].mean())
    X_inseason = transform_inseason(df, season_end_table, global_success_rate, seasons_range).reset_index(drop=True)
    print(f"  {X_inseason.shape[1]}피처  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[3] platoon 피처 (2개, K=520)...", flush=True)
    prior_rate = get_prior_rate(df, season_end_table, global_success_rate, seasons_range)
    platoon_table = build_platoon_table(df)
    X_platoon = transform_platoon(df, platoon_table, prior_rate, seasons_range, k=K_PLATOON).reset_index(drop=True)
    print(f"  {X_platoon.shape[1]}피처  플래툰 셀={len(platoon_table):,}  ({time.time()-t0:.0f}s)", flush=True)

    X_full = pd.concat([X_base, X_inseason, X_platoon], axis=1)
    print(f"\n최종 피처 수={X_full.shape[1]}", flush=True)

    print("\n[4] RF 학습...", flush=True)
    rf = RandomForestClassifier(**RF_PARAMS).fit(X_full, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    print("\n[5] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_full, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(rf)
    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    common = {
        "rf": rf, "hgb": hgb,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(season_end_table, global_success_rate, seasons_range),
        "platoon_stats": platoon_export_stats(platoon_table, seasons_range, k=K_PLATOON),
        "feature_order": list(X_full.columns),
    }

    for tag, w_rf, w_hgb in [("v7a", 0.15, 0.85), ("v7b", 0.0, 1.0)]:
        artifacts = dict(common, w_rf=w_rf, w_hgb=w_hgb)
        out_path = os.path.join(OUT_DIR, f"model_artifacts_{tag}.pkl")
        joblib.dump(artifacts, out_path)
        print(f"저장: {out_path} ({os.path.getsize(out_path)/1e6:.1f}MB) w_rf={w_rf} w_hgb={w_hgb}", flush=True)

    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
