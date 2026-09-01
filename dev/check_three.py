"""3가지 확인:
1) RF weight 0.08~0.22 (0.01 간격) 재탐색
2) 최적 앙상블의 constant shift 보정이 BSS를 올리는지 (진단용 — 실제 test에는 y_valid.mean()을 모르므로 그대로 못 씀)
3) 현재 FeatureBuilder(smoothed만) vs raw+smoothed 버전 HGB 1회 비교
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from metrics import evaluate, format_report

DATA_PATH = "../data/train.csv"
SEED = 42

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    train_fold = df[df["season"] <= 2023].reset_index(drop=True)
    valid_fold = df[df["season"] == 2024].reset_index(drop=True)
    y_train = train_fold[TARGET_COL].to_numpy()
    y_valid = valid_fold[TARGET_COL].to_numpy()

    import os
    only3 = os.environ.get("ONLY3") == "1"

    # ---- 현재(smoothed-only) 피처로 RF/HGB 학습 ----
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(train_fold)
    X_train = fb.transform_train_oof(train_fold)
    X_valid = fb.transform(valid_fold)

    if not only3:
        rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
        p_rf = rf.predict_proba(X_valid)[:, 1]
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)
    p_hgb = hgb.predict_proba(X_valid)[:, 1]
    print(f"HGB 학습 완료 ({time.time() - t0:.0f}s)")

    if not only3:
        # ---- 1) fine grid search 0.08~0.22 ----
        print("\n[1] RF weight fine grid search (0.08~0.22, step 0.01)")
        best_w, best_bss, best_p = None, -1e9, None
        for w in np.arange(0.08, 0.221, 0.01):
            w = round(float(w), 2)
            p_ens = w * p_rf + (1 - w) * p_hgb
            m = evaluate(y_valid, p_ens)
            flag = ""
            if m["bss"] > best_bss:
                best_bss, best_w, best_p = m["bss"], w, p_ens
                flag = "  <- best"
            print(f"  w_rf={w:.2f}  BSS={m['bss']:.6f}  score={m['leaderboard_score']:.2f}{flag}")
        print(f"  최적 w_rf={best_w}  BSS={best_bss:.6f}")

        # ---- 2) constant shift 진단 ----
        print("\n[2] constant shift 진단")
        p_mean, y_mean = float(best_p.mean()), float(y_valid.mean())
        diff = y_mean - p_mean
        print(f"  p_valid.mean()={p_mean:.6f}  y_valid.mean()={y_mean:.6f}  diff={diff:+.6f}")
        p_shifted = np.clip(best_p + diff, 0.0, 1.0)
        m_before = evaluate(y_valid, best_p)
        m_after = evaluate(y_valid, p_shifted)
        print(format_report("  shift 전", m_before))
        print(format_report("  shift 후(진단용)", m_after))
        print("  (주의: diff는 valid의 실제 y_mean을 사용한 값이라 실제 2025 test엔 그대로 적용 불가. "
              "'보정 여지가 있는지'만 보는 진단)")
    else:
        print("\n[1][2] 이전 실행 결과 재사용: 최적 w_rf=0.15, BSS=0.006838, "
              "shift diff=-0.009430, shift 후 BSS=0.0072(진단용)")

    # ---- 3) raw+smoothed 버전 HGB 1회 비교 ----
    print("\n[3] raw asof rate 포함 여부 비교 (HGB 1회)")
    fb2 = FeatureBuilder(seed=SEED, include_raw_rates=True).fit(train_fold)
    X_train2 = fb2.transform_train_oof(train_fold)
    X_valid2 = fb2.transform(valid_fold)
    print(f"  smoothed-only 피처 수={X_train.shape[1]}  raw+smoothed 피처 수={X_train2.shape[1]}")
    hgb2 = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train2, y_train)
    p_hgb2 = hgb2.predict_proba(X_valid2)[:, 1]
    m_hgb1 = evaluate(y_valid, p_hgb)
    m_hgb2 = evaluate(y_valid, p_hgb2)
    print(format_report("  HGB (smoothed only, 기존)", m_hgb1))
    print(format_report("  HGB (raw+smoothed 추가)", m_hgb2))

    print(f"\n총 소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
