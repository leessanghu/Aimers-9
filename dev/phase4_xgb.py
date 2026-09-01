"""Phase 4-2: XGBoost (hist) — logistic vs L2(clip), engineered 피처 공정 비교 + raw ID categorical 별도 트랙.

트랙:
  xgb_log_D   : D 피처셋(pruned 23 + count_asof_ball/diff_prev1_prev5, team_te 제거), binary:logistic
  xgb_l2_D    : 같은 피처셋, reg:squarederror + clip(0,1)
  xgb_log_ids : D 피처셋 + pitcher_id/batter_id raw categorical, binary:logistic
  xgb_l2_ids  : 같은 구성, reg:squarederror + clip

- 1차는 고정 파라미터 비교 (LGBM 튜닝값의 정신을 따른 보수적 설정). 승자만 이후 Optuna 재튜닝.
- early stopping: train 내부 시간순 마지막 8% (valid 연도 정답 미사용, 기존 프로토콜 동일)
출력: dev/phase4_preds/fold_{season}_xgb_variants.csv + summary/sensitivity CSV
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import xgboost as xgb

from phase2_common import FOLDS, build_fold, load_full, rich_eval, time_split_es, all_weight_sensitivity

SEED = 42
OUT_DIR = "phase4_preds"

EXTRA_FEATURES = {"count_asof_ball", "diff_prev1_prev5"}
DEAD_LIST_EXCL_SEASON = [
    c for c in pd.read_csv("dead_features_conservative_list.csv")["feature"].tolist()
    if c != "season"
]

BASE_PARAMS = dict(
    n_estimators=3000, learning_rate=0.01, max_depth=8, min_child_weight=20,
    subsample=0.9, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
    max_bin=256, tree_method="hist", random_state=SEED, n_jobs=-1,
    early_stopping_rounds=100,
)

TRACKS = {
    "xgb_log_D": {"objective": "binary:logistic", "ids": False},
    "xgb_l2_D": {"objective": "reg:squarederror", "ids": False},
    "xgb_log_ids": {"objective": "binary:logistic", "ids": True},
    "xgb_l2_ids": {"objective": "reg:squarederror", "ids": True},
}


def prep_matrices(fold, use_ids):
    X_train = fold["X_train"].drop(columns=[c for c in DEAD_LIST_EXCL_SEASON
                                            if c in fold["X_train"].columns]).copy()
    X_valid = fold["X_valid"].drop(columns=[c for c in DEAD_LIST_EXCL_SEASON
                                            if c in fold["X_valid"].columns]).copy()
    # FeatureBuilder가 만든 category 컬럼(cat_*, count_state, hand_matchup)의 카테고리 값이
    # float(OrdinalEncoder 출력)라 XGBoost가 거부함 -> int로 캐스팅 후 category로 재설정.
    # valid에만 있는 미지 카테고리(NaN)는 -1(OrdinalEncoder의 unknown 관례)로 채운다.
    for c in X_train.columns:
        if str(X_train[c].dtype) == "category":
            tr_int = X_train[c].astype(np.float64).fillna(-1).astype(np.int64)
            va_int = X_valid[c].astype(np.float64).fillna(-1).astype(np.int64)
            all_cats = sorted(set(tr_int.unique()) | set(va_int.unique()))
            X_train[c] = tr_int.astype(pd.CategoricalDtype(categories=all_cats))
            X_valid[c] = va_int.astype(pd.CategoricalDtype(categories=all_cats))
    if use_ids:
        # raw ID를 category dtype으로 — XGBoost native categorical split 사용
        tr_p = fold["train_fold"]["pitcher_id"].astype("category")
        X_train["pitcher_id"] = tr_p
        X_train["batter_id"] = fold["train_fold"]["batter_id"].astype("category")
        X_valid["pitcher_id"] = fold["valid_fold"]["pitcher_id"].astype(
            pd.CategoricalDtype(categories=tr_p.cat.categories))
        X_valid["batter_id"] = fold["valid_fold"]["batter_id"].astype(
            pd.CategoricalDtype(categories=X_train["batter_id"].cat.categories))
    return X_train, X_valid


def fit_predict(X_train, y_train, X_valid, objective, use_ids):
    tr_idx, es_idx = time_split_es(len(X_train))
    is_reg = objective.startswith("reg")
    cls = xgb.XGBRegressor if is_reg else xgb.XGBClassifier
    # cat_*/count_state/hand_matchup가 이미 category dtype이라 모든 트랙에서 필요
    # (LGBM도 동일 컬럼을 categorical로 사용했으므로 공정 비교 유지)
    model = cls(objective=objective, enable_categorical=True, **BASE_PARAMS)
    y_fit = y_train.astype(np.float64) if is_reg else y_train
    model.fit(X_train.iloc[tr_idx], y_fit[tr_idx],
              eval_set=[(X_train.iloc[es_idx], y_fit[es_idx])], verbose=False)
    if is_reg:
        pred = np.clip(model.predict(X_valid), 0.0, 1.0)
    else:
        pred = model.predict_proba(X_valid)[:, 1]
    return pred, model.best_iteration


def main():
    t0 = time.time()
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_full()

    fold_bss = {k: {} for k in TRACKS}
    all_rows = []

    for train_max, valid_season in FOLDS:
        print(f"\n===== fold: train<=season{train_max} -> valid=season{valid_season} =====", flush=True)
        fold = build_fold(df, train_max, valid_season, extra_features=EXTRA_FEATURES, seed=SEED,
                          include_team_te=False)
        preds_df = pd.DataFrame({"row_id": fold["row_id"], "y_valid": fold["y_valid"]})

        for name, track in TRACKS.items():
            tm = time.time()
            X_train, X_valid = prep_matrices(fold, track["ids"])
            pred, best_iter = fit_predict(X_train, fold["y_train"], X_valid,
                                          track["objective"], track["ids"])
            preds_df[f"pred_{name}"] = pred
            m = rich_eval(fold["y_valid"], pred, fold["seen_pitcher_mask"], fold["seen_batter_mask"])
            fold_bss[name][valid_season] = m["bss"]
            m.update({"config": name, "n_features": X_train.shape[1], "valid_season": valid_season,
                      "best_iter": best_iter, "sec": round(time.time() - tm)})
            all_rows.append(m)
            print(f"  [{name}] n_feat={X_train.shape[1]:3d}  BSS={m['bss']:.6f}  score={m['score']:.1f}  "
                  f"calib={m['calib_diff']:+.5f}  seen_p={m['pitcher_seen_bss']:.5f}  "
                  f"unseen_p={m['pitcher_unseen_bss']:.5f}  iters={best_iter}  "
                  f"({time.time()-tm:.0f}s)", flush=True)

        preds_df.to_csv(os.path.join(OUT_DIR, f"fold_{valid_season}_xgb_variants.csv"), index=False)

    summary = pd.DataFrame(all_rows)
    summary.to_csv(os.path.join(OUT_DIR, "phase4_xgb_summary.csv"), index=False, encoding="utf-8")

    print("\n===== 가중치 민감도 =====", flush=True)
    sens_rows = []
    for name in TRACKS:
        sens = all_weight_sensitivity(fold_bss[name])
        sens["config"] = name
        sens.update({f"bss_{s}": v for s, v in fold_bss[name].items()})
        sens_rows.append(sens)
        print(f"  [{name}] " + "  ".join(f"{k}={v:.6f}" for k, v in sens.items() if k != "config"),
              flush=True)
    pd.DataFrame(sens_rows).to_csv(os.path.join(OUT_DIR, "phase4_xgb_sensitivity.csv"),
                                   index=False, encoding="utf-8")
    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
