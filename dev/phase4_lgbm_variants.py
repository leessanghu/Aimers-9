"""Phase 4-1: 튜닝된 LGBM L2 파라미터로 4개 피처 구성(A/B/C/D)을 3폴드 통합 재검증.

배경: phase2_optuna.py는 extra_features=None, include_team_te=True(구식 피처셋)로 돌았음.
     즉 0.004549는 "58피처+team_te" 기준이고, 이후 채택한 피처 변경들과 결합된 적이 없다.
     여기서 같은 튜닝 파라미터로 4개 구성을 공정 비교하고 OOF를 저장한다.

구성:
  A) full 58피처 + team_te 포함   (2024 연장형)
  B) full 58피처 + team_te 제거
  C) pruned 23피처(35개 제거) + team_te 제거   (2023 방어형)
  D) pruned + 신규 2개(count_asof_ball, diff_prev1_prev5) + team_te 제거

출력: dev/phase4_preds/fold_{season}_lgbm_variants.csv (row_id, y_valid, pred_A..pred_D)
      dev/phase4_preds/phase4_lgbm_variants_summary.csv
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation

from phase2_common import FOLDS, build_fold, load_full, rich_eval, time_split_es, all_weight_sensitivity

SEED = 42
OUT_DIR = "phase4_preds"

# phase2_optuna lgbm_l2 최종 선택(candidate 0) 파라미터 + 고정 파라미터
TUNED_L2_PARAMS = dict(
    num_leaves=64, max_depth=12, learning_rate=0.005571638320335239,
    min_child_samples=28, subsample=0.9017762093981382,
    colsample_bytree=0.5291780969405919, reg_alpha=0.07089938907781941,
    reg_lambda=0.009306216375166584, min_split_gain=0.4888649495163153, max_bin=127,
    n_estimators=3000, subsample_freq=1, random_state=SEED, n_jobs=-1, verbosity=-1,
)

DEAD_LIST_EXCL_SEASON = [
    c for c in pd.read_csv("dead_features_conservative_list.csv")["feature"].tolist()
    if c != "season"
]

CONFIGS = {
    "A": {"extra": None, "team_te": True, "drop": []},
    "B": {"extra": None, "team_te": False, "drop": []},
    "C": {"extra": None, "team_te": False, "drop": DEAD_LIST_EXCL_SEASON},
    "D": {"extra": {"count_asof_ball", "diff_prev1_prev5"}, "team_te": False,
          "drop": DEAD_LIST_EXCL_SEASON},
}


def fit_predict(X_train, y_train, X_valid):
    tr_idx, es_idx = time_split_es(len(X_train))
    m = LGBMRegressor(**TUNED_L2_PARAMS)
    m.fit(X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64),
         eval_set=[(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64))], eval_metric="l2",
         callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
    return np.clip(m.predict(X_valid), 0.0, 1.0), m.best_iteration_


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_full()

    fold_bss = {k: {} for k in CONFIGS}
    all_rows = []

    for train_max, valid_season in FOLDS:
        print(f"\n===== fold: train<=season{train_max} -> valid=season{valid_season} =====", flush=True)
        preds_df = None

        for key, cfg in CONFIGS.items():
            tm = time.time()
            fold = build_fold(df, train_max, valid_season, extra_features=cfg["extra"], seed=SEED,
                              include_team_te=cfg["team_te"])
            X_train = fold["X_train"].drop(columns=[c for c in cfg["drop"] if c in fold["X_train"].columns])
            X_valid = fold["X_valid"].drop(columns=[c for c in cfg["drop"] if c in fold["X_valid"].columns])

            if preds_df is None:
                preds_df = pd.DataFrame({"row_id": fold["row_id"], "y_valid": fold["y_valid"]})

            pred, best_iter = fit_predict(X_train, fold["y_train"], X_valid)
            preds_df[f"pred_{key}"] = pred

            m = rich_eval(fold["y_valid"], pred, fold["seen_pitcher_mask"], fold["seen_batter_mask"])
            fold_bss[key][valid_season] = m["bss"]
            m.update({"config": key, "n_features": X_train.shape[1], "valid_season": valid_season,
                      "best_iter": best_iter, "sec": round(time.time() - tm)})
            all_rows.append(m)
            print(f"  [{key}] n_feat={X_train.shape[1]:3d}  BSS={m['bss']:.6f}  score={m['score']:.1f}  "
                  f"calib={m['calib_diff']:+.5f}  seen_p={m['pitcher_seen_bss']:.5f}  "
                  f"unseen_p={m['pitcher_unseen_bss']:.5f}  iters={best_iter}  "
                  f"({time.time()-tm:.0f}s)", flush=True)

        preds_df.to_csv(os.path.join(OUT_DIR, f"fold_{valid_season}_lgbm_variants.csv"), index=False)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(OUT_DIR, "phase4_lgbm_variants_summary.csv"), index=False, encoding="utf-8")

    print("\n===== 가중치 민감도 =====", flush=True)
    sens_rows = []
    for key in CONFIGS:
        sens = all_weight_sensitivity(fold_bss[key])
        sens["config"] = key
        sens.update({f"bss_{s}": v for s, v in fold_bss[key].items()})
        sens_rows.append(sens)
        print(f"  [{key}] " + "  ".join(f"{k}={v:.6f}" for k, v in sens.items() if k != "config"), flush=True)
    pd.DataFrame(sens_rows).to_csv(os.path.join(OUT_DIR, "phase4_weight_sensitivity.csv"),
                                   index=False, encoding="utf-8")
    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
