"""Phase 2 - 3: LGBM classifier / LGBM L2 regressor Optuna 튜닝.

전략 (계산 시간 절약): 2024 fold(가장 큰 fold, train=2019-2023)로 1차 screening
(25~30 trial) -> 상위 5개 후보만 3-fold(2022/2023/2024) 전체로 재평가해서
가중 BSS(0.2/0.3/0.5) 기준 최종 선택.

early stopping은 각 fold의 train 내부 마지막 8%(시간순)만 사용 — valid 정답 사용 안 함.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import optuna
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

from metrics import evaluate
from phase2_common import FOLDS, build_fold, load_full, time_split_es, weighted_bss, FOLD_WEIGHTS

SEED = 42
N_SCREEN_TRIALS = 28
N_TOP_CANDIDATES = 5
SCREEN_TRAIN_MAX, SCREEN_VALID = 2023, 2024  # 가장 큰 fold로 screening

optuna.logging.set_verbosity(optuna.logging.WARNING)

# screening과 top-candidate 재평가에서 반드시 동일해야 하는 고정 파라미터.
# trial.params에는 suggest_*로 튜닝한 값만 들어가므로, 재구성 시 이 dict를 항상 병합해야
# subsample_freq 등 고정값이 재평가 단계에서 빠지는 실수를 막는다.
FIXED_PARAMS = dict(n_estimators=3000, subsample_freq=1, random_state=SEED, n_jobs=-1, verbosity=-1)


def sample_params(trial):
    tuned = dict(
        num_leaves=trial.suggest_int("num_leaves", 15, 255, log=True),
        max_depth=trial.suggest_categorical("max_depth", [-1, 4, 6, 8, 10, 12]),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        min_child_samples=trial.suggest_int("min_child_samples", 20, 500, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        min_split_gain=trial.suggest_float("min_split_gain", 0.0, 1.0),
        max_bin=trial.suggest_categorical("max_bin", [63, 127, 255, 511]),
    )
    return {**tuned, **FIXED_PARAMS}


def fit_eval_cls(params, fold):
    X_train, y_train = fold["X_train"], fold["y_train"]
    tr_idx, es_idx = time_split_es(len(X_train))
    m = LGBMClassifier(**params)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
         eval_set=[(X_train.iloc[es_idx], y_train[es_idx])], eval_metric="binary_logloss",
         callbacks=[early_stopping(80, verbose=False), log_evaluation(0)])
    p = m.predict_proba(fold["X_valid"])[:, 1]
    return evaluate(fold["y_valid"], p)["bss"]


def fit_eval_l2(params, fold):
    X_train, y_train = fold["X_train"], fold["y_train"]
    tr_idx, es_idx = time_split_es(len(X_train))
    m = LGBMRegressor(**params)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64),
         eval_set=[(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64))], eval_metric="l2",
         callbacks=[early_stopping(80, verbose=False), log_evaluation(0)])
    p = np.clip(m.predict(fold["X_valid"]), 0.0, 1.0)
    return evaluate(fold["y_valid"], p)["bss"]


def run_track(name, fit_eval_fn, df, screen_fold):
    print(f"\n########## {name}: screening ({N_SCREEN_TRIALS} trial, fold={SCREEN_VALID}) ##########", flush=True)
    t0 = time.time()

    def objective(trial):
        params = sample_params(trial)
        return fit_eval_fn(params, screen_fold)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=N_SCREEN_TRIALS, show_progress_bar=False)
    print(f"  screening 완료 ({time.time()-t0:.0f}s)  best(2024 only) BSS={study.best_value:.6f}", flush=True)

    top = sorted(study.trials, key=lambda t: t.value if t.value is not None else -1e9, reverse=True)[:N_TOP_CANDIDATES]

    print(f"  상위 {N_TOP_CANDIDATES}개 후보 3-fold 전체 재평가...", flush=True)
    candidate_rows = []
    for i, trial in enumerate(top):
        # trial.params엔 suggest_*로 튜닝한 값만 있으므로 FIXED_PARAMS(subsample_freq 포함)를
        # 반드시 병합해야 screening 때와 동일한 파라미터 구조가 된다.
        params = {**trial.params, **FIXED_PARAMS}
        fold_bss = {}
        tc = time.time()
        for train_max, valid_season in FOLDS:
            fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED)
            fold_bss[valid_season] = fit_eval_fn(params, fold)
        wbss = {wname: weighted_bss(fold_bss, w) for wname, w in FOLD_WEIGHTS.items()}
        row = {"candidate": i, "screen_bss_2024": trial.value, **fold_bss, **wbss, "params": trial.params}
        candidate_rows.append(row)
        print(f"  candidate {i}: fold_bss={fold_bss}  weighted={wbss}  ({time.time()-tc:.0f}s)", flush=True)

    cand_df = pd.DataFrame(candidate_rows).sort_values("default_0.2_0.3_0.5", ascending=False)
    cand_df.to_csv(f"phase2_optuna_{name}_candidates.csv", index=False, encoding="utf-8")
    best = cand_df.iloc[0]
    print(f"  {name} 최종 선택: candidate {best['candidate']}  "
          f"weighted(default)={best['default_0.2_0.3_0.5']:.6f}  params={best['params']}", flush=True)
    print(f"  {name} 트랙 총 소요 {time.time()-t0:.0f}s", flush=True)
    return best


def main():
    t0 = time.time()
    df = load_full()
    print("screening fold 준비 중...", flush=True)
    screen_fold = build_fold(df, SCREEN_TRAIN_MAX, SCREEN_VALID, extra_features=None, seed=SEED)
    print(f"준비 완료 ({time.time()-t0:.0f}s)", flush=True)

    best_cls = run_track("lgbm_cls", fit_eval_cls, df, screen_fold)
    best_l2 = run_track("lgbm_l2", fit_eval_l2, df, screen_fold)

    print("\n===== Phase 2 Optuna 최종 =====", flush=True)
    print(f"lgbm_cls best weighted(default) = {best_cls['default_0.2_0.3_0.5']:.6f}", flush=True)
    print(f"lgbm_l2  best weighted(default) = {best_l2['default_0.2_0.3_0.5']:.6f}", flush=True)
    print(f"총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
