"""ExtraTrees 제거. RF+HGB 2-model weighted ensemble grid search (2019-2023 -> 2024).

RF weight w in [0.0, 0.05, ..., 0.4] (HGB weight = 1-w) 중 최적을 찾고,
HGB 단독보다 개선이 없으면 HGB 단독을 최종으로 채택.
"""

import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL
from metrics import evaluate, format_report

DATA_PATH = "../data/train.csv"
SEED = 42


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    train_fold = df[df["season"] <= 2023].reset_index(drop=True)
    valid_fold = df[df["season"] == 2024].reset_index(drop=True)
    y_train = train_fold[TARGET_COL].to_numpy()
    y_valid = valid_fold[TARGET_COL].to_numpy()
    print(f"train_fold={len(train_fold):,}  valid_fold={len(valid_fold):,}")

    fb = FeatureBuilder(seed=SEED).fit(train_fold)
    X_train = fb.transform_train_oof(train_fold)
    X_valid = fb.transform(valid_fold)
    print(f"피처 수={X_train.shape[1]}  ({time.time() - t0:.0f}s)")

    t1 = time.time()
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
    rf.fit(X_train, y_train)
    p_rf = rf.predict_proba(X_valid)[:, 1]
    print(f"RF 학습 {time.time() - t1:.0f}s")

    t2 = time.time()
    hgb = HistGradientBoostingClassifier(
        max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
        early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
    hgb.fit(X_train, y_train)
    p_hgb = hgb.predict_proba(X_valid)[:, 1]
    print(f"HGB 학습 {time.time() - t2:.0f}s")

    m_rf = evaluate(y_valid, p_rf)
    m_hgb = evaluate(y_valid, p_hgb)
    print("\n" + format_report("RF 단독", m_rf))
    print(format_report("HGB 단독", m_hgb))

    print("\nRF weight grid search:")
    best_w, best_bss = 0.0, m_hgb["bss"]
    for w in np.arange(0.0, 0.41, 0.05):
        w = round(float(w), 2)
        p_ens = w * p_rf + (1 - w) * p_hgb
        m = evaluate(y_valid, p_ens)
        flag = ""
        if m["bss"] > best_bss:
            best_bss, best_w = m["bss"], w
            flag = "  <- best"
        print(f"  w_rf={w:.2f}  BSS={m['bss']:.5f}  score={m['leaderboard_score']:.1f}{flag}")

    print(f"\n최적 w_rf={best_w}  BSS={best_bss:.5f}")
    if best_w == 0.0:
        print("=> HGB 단독보다 나은 RF 혼합 비율 없음. 최종 채택: HGB 단독")
    else:
        print(f"=> 최종 채택: RF {best_w} / HGB {1 - best_w} 가중 앙상블")

    print(f"\n총 소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
