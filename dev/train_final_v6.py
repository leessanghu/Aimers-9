"""최종 제출 모델 v6 = v4(925.908점 실증) + workload 4 + form 4 = 71피처.

v5(코덱스판)는 test.csv 행 순서 기반 rolling을 써서 규칙 위반이었고 폐기됨.
v6의 두 피처군은 전부 '그 행 자신의 컬럼'만 사용한다:
  workload — asof_pitcher_prev1/prev5_game_{success,middle}_rate에 숨은 분모(투구 수) 복원
  form     — asof_pitcher_prev1/3/5_game_success_rate - inseason_success_smooth
in-season은 직전 시즌 종료 시점만 참조. test.csv 행 간 참조 없음.

3폴드 검증(phase8b): rf015_hgb085 delta = 2022 +6.0 / 2023 +85.5 / 2024 +17.1 (전부 양수)
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from inseason import build_season_end_table, transform_inseason, export_stats as inseason_export_stats
from pitchcount_recover import build_workload_features

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
W_RF, W_HGB = 0.15, 0.85

_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")


def strip_rng(obj, seen=None, depth=0):
    """서버 numpy<2.0에서 unpickle 되도록 학습 중 생긴 RNG 객체를 제거."""
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


def build_form_features(df, inseason_success_smooth):
    p1 = df["asof_pitcher_prev1_game_success_rate"].to_numpy(np.float64)
    p3 = df["asof_pitcher_prev3_game_success_rate"].to_numpy(np.float64)
    p5 = df["asof_pitcher_prev5_game_success_rate"].to_numpy(np.float64)
    base = np.asarray(inseason_success_smooth, dtype=np.float64)

    out = pd.DataFrame(index=df.index)
    out["form_prev1"] = np.nan_to_num(p1 - base, nan=0.0)
    out["form_prev3"] = np.nan_to_num(p3 - base, nan=0.0)
    out["form_prev5"] = np.nan_to_num(p5 - base, nan=0.0)
    out["form_trend"] = np.nan_to_num(p1 - p5, nan=0.0)
    return out


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

    print("\n[3] workload 4 + form 4...", flush=True)
    X_workload = build_workload_features(df).reset_index(drop=True)
    X_form = build_form_features(df, X_inseason["inseason_success_smooth"].to_numpy()).reset_index(drop=True)
    q1 = np.expm1(X_workload["recent1_pitch_n"])
    q1 = q1[q1 > 0]
    print(f"  복원 prev1 투구수 median={np.median(q1):.1f}  ({time.time()-t0:.0f}s)", flush=True)

    X_full = pd.concat([X_base, X_inseason, X_workload, X_form], axis=1)
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
    artifacts = {
        "rf": rf, "hgb": hgb, "w_rf": W_RF, "w_hgb": W_HGB,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(season_end_table, global_success_rate, seasons_range),
        "feature_order": list(X_full.columns),
    }
    out_path = os.path.join(OUT_DIR, "model_artifacts_v6.pkl")
    joblib.dump(artifacts, out_path)
    print(f"\n저장: {out_path} ({os.path.getsize(out_path)/1e6:.1f}MB)  총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
