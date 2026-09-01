"""v9 = v7c(948.970 실증, 67피처)에서 batter_asof 4개만 제거 = 63피처.

단일 변수 변경만 한다. v8이 4개를 한꺼번에 바꿔(batter제거+칼만교체+CatBoost+앙상블)
939.875로 떨어졌는데 원인 특정이 불가능했다. 이번엔 제거 하나만 검증한다.
칼만 없음 / CatBoost 없음 / HGB 단독 — 그 외 전부 v7c와 동일.

근거: phase17 ablation 2024폴드에서 -batter_asof = +33.0 (12개 그룹 중 최대 양수).
      타자 정보는 이 타깃(투수가 목표 지점을 맞추는가)에 해로움. 타자 in-season -6.4와도 일관.

--- 이하 v7c 원본 설명 ---
v7c = v7b(939.681 실증: 58 base + 5 in-season + 2 platoon) + inning 2개 피처 = 67피처.
HGB 단독 (RF는 검증 프로토콜상 분산감소 수단이 구조적으로 과대평가된다는 걸 확인해 제외).

phase11_inning.py 3폴드 검증(baseline=v7b 구성, HGB단독):
  2022 +9.2 / 2023 -29.9(원시BSS, baseline이 이미 음수라 실제론 어차피 0으로 floor) / 2024 +19.9

근거: 투수x이닝 상호작용 진짜SD=0.0209(노이즈 제거) -> 상한 ~174점. 전역 이닝 주효과는
      prior에서 빼서 순수 개인 상호작용만 남긴다(모델이 이미 아는 inning 원시피처와 중복 방지).

leakage: inning_table/inning_offset 전부 fit(=train)에서만 계산. 각 행은 자기 투수의
'직전 시즌 끝'까지 누적된 (pitcher, inning) 셀만 조회. 같은 시즌 다른 행 참조 없음.
"""

import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from features import FeatureBuilder, TARGET_COL
from inning_split import build_inning_offset, build_inning_table, transform_inning, K_INNING, export_stats as inning_export_stats
from inseason import build_season_end_table, transform_inseason, _pivots_from_table, export_stats as inseason_export_stats
from platoon import build_platoon_table, transform_platoon, export_stats as platoon_export_stats, K_PLATOON

BATTER_ASOF = ["flag_asof_batter_n_zero", "asof_batter_n",
               "asof_batter_success_rate_smooth", "asof_batter_middle_rate_smooth"]

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
    print(f"  {X_platoon.shape[1]}피처  ({time.time()-t0:.0f}s)", flush=True)

    print("\n[4] inning 피처 (2개, K=570)...", flush=True)
    inning_offset = build_inning_offset(df)
    inning_table = build_inning_table(df)
    X_inning = transform_inning(df, inning_table, inning_offset, prior_rate, seasons_range, k=K_INNING).reset_index(drop=True)
    print(f"  {X_inning.shape[1]}피처  이닝셀={len(inning_table):,}  ({time.time()-t0:.0f}s)", flush=True)

    X_full = pd.concat([X_base, X_inseason, X_platoon, X_inning], axis=1)
    dropped = [c for c in BATTER_ASOF if c in X_full.columns]
    X_full = X_full.drop(columns=dropped)
    print(f"\n제거: {dropped}", flush=True)
    print(f"최종 피처 수={X_full.shape[1]} (v7c 67 -> 63)", flush=True)

    print("\n[5] HGB 학습...", flush=True)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_full, y)
    print(f"  완료 ({time.time()-t0:.0f}s)", flush=True)

    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    artifacts = {
        "hgb": hgb, "w_hgb": 1.0,
        "stats": fb.export_stats(),
        "inseason_stats": inseason_export_stats(season_end_table, global_success_rate, seasons_range),
        "platoon_stats": platoon_export_stats(platoon_table, seasons_range, k=K_PLATOON),
        "inning_stats": inning_export_stats(inning_table, inning_offset, seasons_range, k=K_INNING),
        "feature_order": list(X_full.columns),
        "drop_cols": BATTER_ASOF,
    }
    out_path = os.path.join(OUT_DIR, "model_artifacts_v9.pkl")
    joblib.dump(artifacts, out_path)
    print(f"\n저장: {out_path} ({os.path.getsize(out_path)/1e6:.1f}MB)  총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
