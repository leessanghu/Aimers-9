"""Phase 4-3: CatBoost 제대로 튜닝 — raw pitcher_id/batter_id categorical + Optuna.

평가 기준 전환: 3-fold 가중BSS 대신 2024 단일 폴드 BSS를 1차 선택 기준으로 삼는다
(D+A 블렌딩이 3-fold 가중 기준으론 이겼지만 실제 리더보드에서 하락한 것으로 확인됨 —
 2023 방어형 변경들이 실제 2025엔 안 맞았다는 뜻). 2022/2023은 참고용 가드레일로만 출력.

- CatBoost 공식 권장: categorical은 원핫 인코딩하지 말고 raw로 전달, has_time=True로 시간순 보장
- Logloss(classifier) vs RMSE(regressor+clip) 둘 다 비교
- screening: 2024 fold로 25 trial, depth/l2_leaf_reg/random_strength/bagging_temperature/learning_rate 튜닝
- 상위 3개만 2022/2023도 참고로 재평가 (선택 근거로는 안 씀)
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

from metrics import evaluate
from phase2_common import FOLDS, build_fold, load_full, time_split_es

SEED = 42
N_SCREEN_TRIALS = 25
N_TOP_CANDIDATES = 3
SCREEN_TRAIN_MAX, SCREEN_VALID = 2023, 2024  # 1차 기준 = 2024

EXTRA_FEATURES = {"count_asof_ball", "diff_prev1_prev5"}
DEAD_LIST_EXCL_SEASON = [
    c for c in pd.read_csv("dead_features_conservative_list.csv")["feature"].tolist()
    if c != "season"
]
CAT_FEATURES_BASE = ["cat_top_bottom", "cat_game_type", "cat_base_state", "count_state", "hand_matchup"]
ID_COLS = ["pitcher_id", "batter_id"]

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_matrices(fold):
    X_train = fold["X_train"].drop(columns=[c for c in DEAD_LIST_EXCL_SEASON
                                            if c in fold["X_train"].columns]).copy()
    X_valid = fold["X_valid"].drop(columns=[c for c in DEAD_LIST_EXCL_SEASON
                                            if c in fold["X_valid"].columns]).copy()
    X_train["pitcher_id"] = fold["train_fold"]["pitcher_id"].to_numpy()
    X_train["batter_id"] = fold["train_fold"]["batter_id"].to_numpy()
    X_valid["pitcher_id"] = fold["valid_fold"]["pitcher_id"].to_numpy()
    X_valid["batter_id"] = fold["valid_fold"]["batter_id"].to_numpy()

    # cat_top_bottom 등 일부 CAT_FEATURES_BASE 컬럼이 죽은 피처 목록에 있어 이미 드롭됐을 수 있음
    cat_features = [c for c in CAT_FEATURES_BASE if c in X_train.columns] + ID_COLS
    for c in cat_features:
        X_train[c] = X_train[c].astype(np.int64).astype(str)
        X_valid[c] = X_valid[c].astype(np.int64).astype(str)
    return X_train, X_valid, cat_features


def sample_params(trial):
    return dict(
        iterations=3000,
        depth=trial.suggest_int("depth", 4, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
        l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        random_strength=trial.suggest_float("random_strength", 0.0, 3.0),
        bagging_temperature=trial.suggest_float("bagging_temperature", 0.0, 3.0),
        border_count=trial.suggest_categorical("border_count", [64, 128, 254]),
        random_seed=SEED, thread_count=-1, verbose=0, allow_writing_files=False,
        early_stopping_rounds=100, has_time=True,
    )


def fit_eval(params, X_train, y_train, X_valid, y_valid, cat_features, tr_idx, es_idx, loss):
    pool_tr = Pool(X_train.iloc[tr_idx], y_train[tr_idx], cat_features=cat_features)
    pool_es = Pool(X_train.iloc[es_idx], y_train[es_idx], cat_features=cat_features)
    pool_va = Pool(X_valid, cat_features=cat_features)
    if loss == "Logloss":
        m = CatBoostClassifier(loss_function="Logloss", **params)
        m.fit(pool_tr, eval_set=pool_es)
        p = m.predict_proba(pool_va)[:, 1]
    else:
        m = CatBoostRegressor(loss_function="RMSE", **params)
        m.fit(pool_tr, eval_set=pool_es)
        p = np.clip(m.predict(pool_va), 0.0, 1.0)
    return evaluate(y_valid, p)["bss"], m.get_best_iteration()


def run_track(name, loss, df, screen_fold):
    print(f"\n########## {name}: screening ({N_SCREEN_TRIALS} trial, 2024 기준) ##########", flush=True)
    t0 = time.time()
    X_train, X_valid, cat_features = build_matrices(screen_fold)
    y_train, y_valid = screen_fold["y_train"], screen_fold["y_valid"]
    tr_idx, es_idx = time_split_es(len(X_train))

    def objective(trial):
        params = sample_params(trial)
        bss, _ = fit_eval(params, X_train, y_train, X_valid, y_valid, cat_features, tr_idx, es_idx, loss)
        return bss

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_SCREEN_TRIALS, show_progress_bar=False)
    print(f"  screening 완료 ({time.time()-t0:.0f}s)  best(2024) BSS={study.best_value:.6f}", flush=True)

    top = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1e9,
                reverse=True)[:N_TOP_CANDIDATES]
    print(f"  상위 {N_TOP_CANDIDATES}개 후보 2022/2023 참고 평가...", flush=True)
    rows = []
    for i, trial in enumerate(top):
        params = {**trial.params, "iterations": 3000, "random_seed": SEED, "thread_count": -1,
                  "verbose": 0, "allow_writing_files": False, "early_stopping_rounds": 100,
                  "has_time": True}
        row = {"candidate": i, "bss_2024": trial.value, "params": trial.params}
        for train_max, valid_season in FOLDS:
            if valid_season == 2024:
                continue
            fold = build_fold(df, train_max, valid_season, extra_features=EXTRA_FEATURES, seed=SEED,
                              include_team_te=False)
            Xt, Xv, cf = build_matrices(fold)
            tri, esi = time_split_es(len(Xt))
            bss, _ = fit_eval(params, Xt, fold["y_train"], Xv, fold["y_valid"], cf, tri, esi, loss)
            row[f"bss_{valid_season}"] = bss
        rows.append(row)
        print(f"  candidate {i}: {row}", flush=True)

    cand_df = pd.DataFrame(rows).sort_values("bss_2024", ascending=False)
    cand_df.to_csv(f"phase4_catboost_{name}_candidates.csv", index=False, encoding="utf-8")
    best = cand_df.iloc[0]
    print(f"  {name} 최종(2024 기준) 선택: candidate {best['candidate']}  bss_2024={best['bss_2024']:.6f}  "
          f"params={best['params']}", flush=True)
    print(f"  {name} 총 소요 {time.time()-t0:.0f}s", flush=True)
    return best


def main():
    t0 = time.time()
    df = load_full()
    screen_fold = build_fold(df, SCREEN_TRAIN_MAX, SCREEN_VALID, extra_features=EXTRA_FEATURES, seed=SEED,
                             include_team_te=False)

    best_log = run_track("logloss", "Logloss", df, screen_fold)
    best_rmse = run_track("rmse", "RMSE", df, screen_fold)

    print("\n===== CatBoost Optuna 최종 (2024 기준) =====", flush=True)
    print(f"logloss best bss_2024 = {best_log['bss_2024']:.6f}", flush=True)
    print(f"rmse    best bss_2024 = {best_rmse['bss_2024']:.6f}", flush=True)
    print(f"총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
