"""Phase 1: CatBoost(raw ID categorical) vs LGBM classifier vs LGBM L2 regressor,
각각 recency weight 4단계 (없음/약/중/강) — 2019-2023 학습 -> 2024 검증.

- 교차 범주(count_state, hand_matchup)는 features.py에 추가됨
- CatBoost: pitcher_id/batter_id를 raw categorical로 직접 사용
- LGBM L2: 예측을 [0,1]로 clip (y가 0/1 + squared error에서 clip은 Brier를 절대 악화시키지 않음)
  clip 전 이탈 비율/크기 기록, 1% 이상 이탈 시 불안정 판정
- early stopping: train 내부 시간순 마지막 8%를 eval_set으로 (2024는 절대 사용 안 함)
- recency weight: w = decay^(2023 - season), decay ∈ {1.0, 0.9, 0.75, 0.55}
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping, log_evaluation

from features import FeatureBuilder, TARGET_COL
from metrics import evaluate

DATA_PATH = "../data/train.csv"
SEED = 42
DECAYS = {"none": 1.0, "weak": 0.9, "mid": 0.75, "strong": 0.55}
ES_FRAC = 0.08  # train 내부 early-stopping 홀드아웃 (시간순 마지막 8%)

CAT_FEATURES_COMMON = ["cat_top_bottom", "cat_game_type", "cat_base_state",
                       "count_state", "hand_matchup"]
CAT_FEATURES_CB = CAT_FEATURES_COMMON + ["pitcher_id", "batter_id"]


def season_weights(seasons, decay):
    return np.power(decay, (2023 - seasons).clip(lower=0)).astype(np.float64)


def time_split_es(n):
    """행 순서(=시간순)를 그대로 이용해 마지막 ES_FRAC를 early-stopping용으로 분리."""
    cut = int(n * (1 - ES_FRAC))
    return np.arange(cut), np.arange(cut, n)


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    train_fold = df[df["season"] <= 2023].reset_index(drop=True)
    valid_fold = df[df["season"] == 2024].reset_index(drop=True)
    y_train = train_fold[TARGET_COL].to_numpy()
    y_valid = valid_fold[TARGET_COL].to_numpy()
    seasons = train_fold["season"]

    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(train_fold)
    X_train = fb.transform_train_oof(train_fold)
    X_valid = fb.transform(valid_fold)

    # CatBoost용: raw ID 추가
    X_train_cb = X_train.copy()
    X_valid_cb = X_valid.copy()
    X_train_cb["pitcher_id"] = train_fold["pitcher_id"].to_numpy()
    X_train_cb["batter_id"] = train_fold["batter_id"].to_numpy()
    X_valid_cb["pitcher_id"] = valid_fold["pitcher_id"].to_numpy()
    X_valid_cb["batter_id"] = valid_fold["batter_id"].to_numpy()

    # categorical 컬럼은 정수형으로 (CatBoost/LGBM 공통)
    for c in CAT_FEATURES_CB:
        X_train_cb[c] = X_train_cb[c].astype(np.int64)
        X_valid_cb[c] = X_valid_cb[c].astype(np.int64)
    for c in CAT_FEATURES_COMMON:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype(pd.CategoricalDtype(categories=X_train[c].cat.categories))

    tr_idx, es_idx = time_split_es(len(X_train))
    print(f"train={len(tr_idx):,}  es_holdout={len(es_idx):,}  valid2024={len(y_valid):,}  "
          f"features(lgbm)={X_train.shape[1]}  features(cb)={X_train_cb.shape[1]}  ({time.time()-t0:.0f}s)",
          flush=True)

    results = []

    for wname, decay in DECAYS.items():
        w = season_weights(seasons, decay)
        w_tr, w_es = w[tr_idx], w[es_idx]

        # ---------- LGBM classifier ----------
        tm = time.time()
        lgb_cls = LGBMClassifier(
            n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
            min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)
        lgb_cls.fit(
            X_train.iloc[tr_idx], y_train[tr_idx], sample_weight=w_tr,
            eval_set=[(X_train.iloc[es_idx], y_train[es_idx])], eval_sample_weight=[w_es],
            eval_metric="binary_logloss",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        p = lgb_cls.predict_proba(X_valid)[:, 1]
        m = evaluate(y_valid, p)
        results.append({"model": "lgbm_cls", "weight": wname, "bss": m["bss"],
                        "score": m["leaderboard_score"], "best_iter": lgb_cls.best_iteration_,
                        "sec": round(time.time() - tm)})
        print(f"lgbm_cls  w={wname:6s}  BSS={m['bss']:.6f}  score={m['leaderboard_score']:.1f}  "
              f"iters={lgb_cls.best_iteration_}  ({time.time()-tm:.0f}s)", flush=True)

        # ---------- LGBM L2 regressor ----------
        tm = time.time()
        lgb_l2 = LGBMRegressor(
            n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
            min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
            colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)
        lgb_l2.fit(
            X_train.iloc[tr_idx], y_train[tr_idx].astype(np.float64), sample_weight=w_tr,
            eval_set=[(X_train.iloc[es_idx], y_train[es_idx].astype(np.float64))],
            eval_sample_weight=[w_es], eval_metric="l2",
            callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        p_raw = lgb_l2.predict(X_valid)
        out_lo, out_hi = (p_raw < 0).mean(), (p_raw > 1).mean()
        out_frac = out_lo + out_hi
        p = np.clip(p_raw, 0.0, 1.0)
        m = evaluate(y_valid, p)
        results.append({"model": "lgbm_l2", "weight": wname, "bss": m["bss"],
                        "score": m["leaderboard_score"], "best_iter": lgb_l2.best_iteration_,
                        "out_frac": round(float(out_frac), 5),
                        "out_min": round(float(p_raw.min()), 4), "out_max": round(float(p_raw.max()), 4),
                        "pred_mean": round(float(p.mean()), 5),
                        "sec": round(time.time() - tm)})
        print(f"lgbm_l2   w={wname:6s}  BSS={m['bss']:.6f}  score={m['leaderboard_score']:.1f}  "
              f"iters={lgb_l2.best_iteration_}  clip이탈={out_frac*100:.3f}% "
              f"[{p_raw.min():.3f},{p_raw.max():.3f}]  ({time.time()-tm:.0f}s)", flush=True)

        # ---------- CatBoost (raw ID categorical) ----------
        tm = time.time()
        cb = CatBoostClassifier(
            iterations=3000, learning_rate=0.06, depth=6, l2_leaf_reg=5.0,
            loss_function="Logloss", random_seed=SEED, verbose=0,
            early_stopping_rounds=100, thread_count=-1, allow_writing_files=False)
        pool_tr = Pool(X_train_cb.iloc[tr_idx], y_train[tr_idx], weight=w_tr,
                       cat_features=CAT_FEATURES_CB)
        pool_es = Pool(X_train_cb.iloc[es_idx], y_train[es_idx], weight=w_es,
                       cat_features=CAT_FEATURES_CB)
        cb.fit(pool_tr, eval_set=pool_es)
        p = cb.predict_proba(Pool(X_valid_cb, cat_features=CAT_FEATURES_CB))[:, 1]
        m = evaluate(y_valid, p)
        results.append({"model": "catboost", "weight": wname, "bss": m["bss"],
                        "score": m["leaderboard_score"], "best_iter": cb.get_best_iteration(),
                        "sec": round(time.time() - tm)})
        print(f"catboost  w={wname:6s}  BSS={m['bss']:.6f}  score={m['leaderboard_score']:.1f}  "
              f"iters={cb.get_best_iteration()}  ({time.time()-tm:.0f}s)", flush=True)

    res = pd.DataFrame(results).sort_values("bss", ascending=False)
    res.to_csv("phase1_results.csv", index=False, encoding="utf-8")
    print("\n===== Phase 1 최종 (BSS 내림차순) =====", flush=True)
    print(res.to_string(index=False), flush=True)
    print(f"\n현재 챔피언(RF0.15+HGB0.85) = BSS 0.006838 / score 683.75", flush=True)
    print(f"총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
