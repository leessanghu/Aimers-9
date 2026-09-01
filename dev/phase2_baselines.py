"""Phase 2 - 1,2: 생존 모델 4종(LGBM cls / LGBM L2 / sklearn HGB / RF, 전부 기존 최선 설정)을
rolling fold(2022/2023/2024) 전체에서 학습·평가하고 예측을 저장한다.
CatBoost는 Phase 1에서 부진 + 느림 확인되어 제외 (나중에 별도 튜닝 대상).
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from phase2_common import FOLDS, build_fold, load_full, rich_eval, time_split_es, all_weight_sensitivity

SEED = 42
OUT_DIR = "phase2_preds"

LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)
RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)


def fit_lgbm_cls(X_train, y_train, tr_idx, es_idx):
    m = LGBMClassifier(**LGBM_PARAMS)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
         eval_set=[(X_train.iloc[es_idx], y_train[es_idx])], eval_metric="binary_logloss",
         callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    return m, lambda X: m.predict_proba(X)[:, 1]


def fit_lgbm_l2(X_train, y_train, tr_idx, es_idx):
    m = LGBMRegressor(**LGBM_PARAMS)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64),
         eval_set=[(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64))], eval_metric="l2",
         callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    return m, lambda X: np.clip(m.predict(X), 0.0, 1.0)


def fit_hgb(X_train, y_train, tr_idx, es_idx):
    m = HistGradientBoostingClassifier(**HGB_PARAMS)
    m.fit(X_train, y_train)  # HGB는 내부 validation_fraction으로 자체 early stopping
    return m, lambda X: m.predict_proba(X)[:, 1]


def fit_rf(X_train, y_train, tr_idx, es_idx):
    m = RandomForestClassifier(**RF_PARAMS)
    m.fit(X_train, y_train)
    return m, lambda X: m.predict_proba(X)[:, 1]


MODELS = {
    "lgbm_cls": fit_lgbm_cls,
    "lgbm_l2": fit_lgbm_l2,
    "hgb": fit_hgb,
    "rf": fit_rf,
}


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_full()

    fold_bss = {name: {} for name in list(MODELS) + ["lgbm_cls_no_team_te"]}
    all_rows = []

    for train_max, valid_season in FOLDS:
        print(f"\n===== fold: train<=season{train_max} -> valid=season{valid_season} =====", flush=True)
        tf = time.time()
        fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED, include_team_te=True)
        X_train, X_valid = fold["X_train"], fold["X_valid"]
        y_train, y_valid = fold["y_train"], fold["y_valid"]
        tr_idx, es_idx = time_split_es(len(X_train))
        print(f"  train={len(tr_idx):,}  es={len(es_idx):,}  valid={len(y_valid):,}  "
              f"features={X_train.shape[1]}  ({time.time()-tf:.0f}s)", flush=True)

        # team_te(shuffled KFold, 시간 구조 안 맞음) 제거 버전 — lgbm_cls 하나로만 비교
        fold_no_te = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED,
                                include_team_te=False)

        preds_df = pd.DataFrame({"row_id": fold["row_id"], "y_valid": y_valid})

        for name, fit_fn in MODELS.items():
            tm = time.time()
            model, predict_fn = fit_fn(X_train, y_train, tr_idx, es_idx)
            pred = predict_fn(X_valid)
            preds_df[f"pred_{name}"] = pred
            m = rich_eval(y_valid, pred, fold["seen_pitcher_mask"], fold["seen_batter_mask"])
            fold_bss[name][valid_season] = m["bss"]
            m.update({"model": name, "valid_season": valid_season, "sec": round(time.time() - tm)})
            all_rows.append(m)
            print(f"  {name:10s} BSS={m['bss']:.6f} score={m['score']:.1f}  "
                  f"calib_diff={m['calib_diff']:+.5f}  "
                  f"seen_p_bss={m['pitcher_seen_bss']:.5f}(n={m['pitcher_seen_n']})  "
                  f"unseen_p_bss={m['pitcher_unseen_bss']:.5f}(n={m['pitcher_unseen_n']})  "
                  f"seen_b_bss={m['batter_seen_bss']:.5f}(n={m['batter_seen_n']})  "
                  f"unseen_b_bss={m['batter_unseen_bss']:.5f}(n={m['batter_unseen_n']})  "
                  f"({time.time()-tm:.0f}s)", flush=True)

        # team_te 제거 버전 (lgbm_cls만) — team_te가 shuffled KFold라 시간 구조와 안 맞을 수 있어
        # 포함/제거를 직접 비교
        tm = time.time()
        tr_idx_nt, es_idx_nt = time_split_es(len(fold_no_te["X_train"]))
        model_nt, predict_fn_nt = fit_lgbm_cls(fold_no_te["X_train"], fold_no_te["y_train"],
                                               tr_idx_nt, es_idx_nt)
        pred_nt = predict_fn_nt(fold_no_te["X_valid"])
        preds_df["pred_lgbm_cls_no_team_te"] = pred_nt
        m_nt = rich_eval(y_valid, pred_nt, fold["seen_pitcher_mask"], fold["seen_batter_mask"])
        fold_bss["lgbm_cls_no_team_te"][valid_season] = m_nt["bss"]
        m_nt.update({"model": "lgbm_cls_no_team_te", "valid_season": valid_season,
                    "sec": round(time.time() - tm)})
        all_rows.append(m_nt)
        print(f"  {'lgbm_cls_no_team_te':10s} BSS={m_nt['bss']:.6f} score={m_nt['score']:.1f}  "
              f"(team_te 포함판 대비 {m_nt['bss']-fold_bss['lgbm_cls'][valid_season]:+.6f})  "
              f"({time.time()-tm:.0f}s)", flush=True)

        preds_df.to_csv(os.path.join(OUT_DIR, f"fold_{valid_season}_preds.csv"), index=False)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(OUT_DIR, "phase2_baseline_summary.csv"), index=False, encoding="utf-8")

    print("\n===== 가중치 민감도 (fold별 BSS 가중합) =====", flush=True)
    sens_rows = []
    for name in fold_bss:
        sens = all_weight_sensitivity(fold_bss[name])
        sens["model"] = name
        sens.update({f"bss_{s}": fold_bss[name][s] for s in fold_bss[name]})
        sens_rows.append(sens)
        print(f"{name:10s}  " + "  ".join(f"{k}={v:.6f}" for k, v in sens.items() if k != "model"), flush=True)
    pd.DataFrame(sens_rows).to_csv(os.path.join(OUT_DIR, "phase2_weight_sensitivity.csv"),
                                   index=False, encoding="utf-8")

    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
