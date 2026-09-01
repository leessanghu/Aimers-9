"""로컬 검증 스크립트.

split      : season 2019-2023 = train_fold, season 2024 = valid_fold (시간 기반)
metric     : Brier Score / Brier Skill Score (EVALUATION.md 공식 그대로, dev/metrics.py)
비교 대상  :
  A) baseline-arch  : baseline_submit과 동일한 파이프라인(median impute + ordinal encode
                       + RF max_depth=10 min_samples_leaf=200)을 train_fold로 재학습
                       -> 우리 검증 프로토콜에서 baseline 아키텍처의 진짜 held-out 성능
  B) rf.pkl(제공됨)  : 제출된 baseline_submit/model/rf.pkl을 그대로 valid_fold에 평가
                       -> 이 모델은 2019~2024 전체로 학습됐을 가능성이 높아 2024가
                          일부 in-sample일 수 있음(참고용, 과대추정 가능)
  C) engineered ens. : dev/features.py로 만든 피처 + RF/ExtraTrees/HGB 가중평균 앙상블
"""

import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from features import FeatureBuilder, TARGET_COL
from metrics import evaluate, format_report

DATA_PATH = "../data/train.csv"
SEED = 42


def load_split():
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    train_fold = df[df["season"] <= 2023].reset_index(drop=True)
    valid_fold = df[df["season"] == 2024].reset_index(drop=True)
    return train_fold, valid_fold


def build_baseline_arch_pipeline():
    """baseline_submit/script.py의 학습 파이프라인 구조를 재현 (동일 검증셋 비교용)."""
    cat_cols = ["top_bottom", "game_type", "base_state"]
    num_cols = [
        "season", "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before",
        "outs_before", "run_top_before", "run_bot_before", "run_total_before", "score_diff_home",
        "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b", "num_runners_on",
        "home_win_expectancy", "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand",
        "batter_hand", "pitcher_team_id", "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
        "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
        "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
        "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
    ]
    pre = ColumnTransformer([
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), cat_cols),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ])
    clf = RandomForestClassifier(max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)]), cat_cols + num_cols


def main():
    t0 = time.time()
    print("데이터 로드 및 시간 기반 split...")
    train_fold, valid_fold = load_split()
    print(f"  train_fold(2019-2023)={len(train_fold):,}  valid_fold(2024)={len(valid_fold):,}")
    y_train = train_fold[TARGET_COL].to_numpy()
    y_valid = valid_fold[TARGET_COL].to_numpy()

    import os
    skip_baseline = os.environ.get("SKIP_BASELINE") == "1"
    results = []

    if not skip_baseline:
        # ---- A) baseline 아키텍처를 train_fold로 재학습 (공정 비교) ----
        print("\n[A] baseline 아키텍처 재학습 (2019-2023 -> 2024 평가)...")
        ta = time.time()
        base_pipe, base_cols = build_baseline_arch_pipeline()
        base_pipe.fit(train_fold[base_cols], y_train)
        p_base = base_pipe.predict_proba(valid_fold[base_cols])[:, 1]
        results.append(("A) baseline-arch (재학습)", evaluate(y_valid, p_base)))
        print(f"  {time.time() - ta:.0f}s")

        # ---- B) 제공된 rf.pkl 그대로 평가 (참고용, in-sample 가능성) ----
        print("\n[B] 제공된 baseline_submit/model/rf.pkl 평가...")
        try:
            provided = joblib.load("../baseline_submit/model/rf.pkl")
            raw_cols = [c for c in valid_fold.columns if c not in ("row_id", TARGET_COL)]
            p_provided = provided.predict_proba(valid_fold[raw_cols])[:, 1]
            results.append(("B) rf.pkl 제공본 (참고, in-sample 우려)", evaluate(y_valid, p_provided)))
        except Exception as e:
            print(f"  건너뜀: {e}")

    # ---- C) 엔지니어링 피처 + 앙상블 ----
    print("\n[C] 피처 엔지니어링 (스무딩/코드스타트/team target encoding)...")
    tc = time.time()
    fb = FeatureBuilder(seed=SEED).fit(train_fold)
    X_train = fb.transform_train_oof(train_fold)
    X_valid = fb.transform(valid_fold)
    print(f"  피처 수={X_train.shape[1]}  {time.time() - tc:.0f}s")

    models = {
        "rf": RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED),
        "et": ExtraTreesClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED),
        "hgb": HistGradientBoostingClassifier(
            max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED),
    }

    preds = {}
    for name, model in models.items():
        tm = time.time()
        model.fit(X_train, y_train)
        p = model.predict_proba(X_valid)[:, 1]
        preds[name] = p
        m = evaluate(y_valid, p)
        results.append((f"C) engineered single:{name}", m))
        print(f"  {name}: {time.time() - tm:.0f}s  BSS={m['bss']:.4f}  score={m['leaderboard_score']:.1f}")

    # 개별 모델 BSS에 비례한 가중치로 확률 평균 (BSS가 낮으면(<=0) 최소 가중치 부여)
    bss_vals = {k: max(evaluate(y_valid, v)["bss"], 1e-4) for k, v in preds.items()}
    total = sum(bss_vals.values())
    weights = {k: v / total for k, v in bss_vals.items()}
    print(f"  ensemble weights (BSS-proportional): {weights}")
    p_ens = sum(weights[k] * preds[k] for k in preds)
    results.append(("C) engineered ensemble (가중평균)", evaluate(y_valid, p_ens)))

    print("\n" + "=" * 100)
    print("최종 리포트 (2024 season 검증)")
    print("=" * 100)
    for name, m in results:
        print(format_report(name, m))
    print(f"\n총 소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
