"""Phase 2 - 5: dead_features.py의 permutation 결과를 '제거 ablation' 성능으로 재검증.

permutation importance는 폴드 내부에서 상수인 컬럼(season 등)을 못 잡아낸다
(섞어도 값이 안 바뀌어서 delta=0). 그래서 여기서는 실제로 컬럼을 지우고 재학습해서
3-fold 가중 BSS가 정말 떨어지는지/안 떨어지는지로 최종 판단한다.

두 버전 비교:
  A) 죽은 피처 목록 36개 전부 제거 (season 포함) — 착시 여부 확인용
  B) season 제외 35개 제거 — 실질적 제거 후보 (season은 사전에 강한 반대 증거가 있어 제외)
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation

from metrics import evaluate
from phase2_common import FOLDS, build_fold, load_full, time_split_es, weighted_bss, FOLD_WEIGHTS

SEED = 42
LGBM_PARAMS = dict(
    n_estimators=3000, learning_rate=0.03, num_leaves=63, max_depth=-1,
    min_child_samples=200, reg_lambda=5.0, subsample=0.9, subsample_freq=1,
    colsample_bytree=0.8, random_state=SEED, n_jobs=-1, verbosity=-1)


def run_drop(df, drop_cols, label):
    fold_bss = {}
    for train_max, valid_season in FOLDS:
        fold = build_fold(df, train_max, valid_season, extra_features=None, seed=SEED)
        X_train = fold["X_train"].drop(columns=[c for c in drop_cols if c in fold["X_train"].columns])
        X_valid = fold["X_valid"].drop(columns=[c for c in drop_cols if c in fold["X_valid"].columns])
        y_train = fold["y_train"]
        tr_idx, es_idx = time_split_es(len(X_train))
        m = LGBMClassifier(**LGBM_PARAMS)
        m.fit(X_train.iloc[tr_idx], y_train[tr_idx],
             eval_set=[(X_train.iloc[es_idx], y_train[es_idx])], eval_metric="binary_logloss",
             callbacks=[early_stopping(100, verbose=False), log_evaluation(0)])
        p = m.predict_proba(X_valid)[:, 1]
        fold_bss[valid_season] = evaluate(fold["y_valid"], p)["bss"]
    wbss = {wname: weighted_bss(fold_bss, w) for wname, w in FOLD_WEIGHTS.items()}
    return {"config": label, "n_features": X_train.shape[1], **fold_bss, **wbss}


def main():
    t0 = time.time()
    df = load_full()

    dead_list = pd.read_csv("dead_features_conservative_list.csv")["feature"].tolist()
    print(f"conservative dead-feature 목록 ({len(dead_list)}개): {dead_list}", flush=True)

    rows = []

    print("\n[baseline] 전체 피처 유지...", flush=True)
    tb = time.time()
    rows.append(run_drop(df, [], "baseline_all_features"))
    print(f"  {rows[-1]}  ({time.time()-tb:.0f}s)", flush=True)

    print(f"\n[A] 36개 전부 제거 (season 포함, 착시 검증용)...", flush=True)
    tb = time.time()
    rows.append(run_drop(df, dead_list, "drop_all_36_incl_season"))
    print(f"  {rows[-1]}  ({time.time()-tb:.0f}s)", flush=True)

    dead_list_no_season = [c for c in dead_list if c != "season"]
    print(f"\n[B] season 제외 {len(dead_list_no_season)}개만 제거 (실질 후보)...", flush=True)
    tb = time.time()
    rows.append(run_drop(df, dead_list_no_season, "drop_35_excl_season"))
    print(f"  {rows[-1]}  ({time.time()-tb:.0f}s)", flush=True)

    result = pd.DataFrame(rows)
    result.to_csv("phase2_dead_feature_removal.csv", index=False, encoding="utf-8")

    base_w = result.loc[result["config"] == "baseline_all_features", "default_0.2_0.3_0.5"].iloc[0]
    print("\n===== 제거 ablation 결과 =====", flush=True)
    print(result.to_string(index=False), flush=True)
    for _, r in result.iterrows():
        if r["config"] != "baseline_all_features":
            print(f"{r['config']}: baseline 대비 가중BSS 변화 = {r['default_0.2_0.3_0.5'] - base_w:+.6f}",
                  flush=True)

    print("\n[해석 가이드]")
    print("  A(season 포함 제거)가 크게 나빠지면 -> season의 0-delta는 예상대로 폴드 내부 상수라 생긴 착시 확정.")
    print("  B(season 제외 제거)가 baseline과 비슷하거나 낫다면 -> 나머지 35개는 실제로 제거해도 안전.")
    print(f"\n총 소요 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
