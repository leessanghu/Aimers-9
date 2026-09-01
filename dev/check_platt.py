"""시간 구조를 살린 Platt calibration 실험.

1) 2019-2022 학습 -> 2023 예측 (RF 0.15 + HGB 0.85 앙상블)
2) 2023 예측(logit) vs 2023 정답으로 LogisticRegression(Platt) 학습
3) 2019-2023 학습 -> 2024 예측 (기존 683.8 재현)
4) 같은 Platt를 2024 예측에 적용
5) 원본 대비 BSS 개선폭 보고. +0.0005 이상이면 채택, 아니면 폐기.
"""

import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from features import FeatureBuilder, TARGET_COL
from metrics import evaluate, format_report

DATA_PATH = "../data/train.csv"
SEED = 42
W_RF = 0.15

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)

ADOPT_THRESHOLD = 0.0005


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def train_ensemble(train_df, valid_df):
    y_train = train_df[TARGET_COL].to_numpy()
    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(train_df)
    X_train = fb.transform_train_oof(train_df)
    X_valid = fb.transform(valid_df)

    rf = RandomForestClassifier(**RF_PARAMS).fit(X_train, y_train)
    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X_train, y_train)

    p_rf = rf.predict_proba(X_valid)[:, 1]
    p_hgb = hgb.predict_proba(X_valid)[:, 1]
    return W_RF * p_rf + (1 - W_RF) * p_hgb


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")

    # ---- Fold1: 2019-2022 -> 2023 (Platt 학습용) ----
    print("[Fold1] 2019-2022 학습 -> 2023 예측...")
    train_a = df[df["season"] <= 2022].reset_index(drop=True)
    valid_a = df[df["season"] == 2023].reset_index(drop=True)
    y_2023 = valid_a[TARGET_COL].to_numpy()
    p_2023 = train_ensemble(train_a, valid_a)
    m_2023_raw = evaluate(y_2023, p_2023)
    print(format_report("  2023 raw ensemble", m_2023_raw))
    print(f"  ({time.time() - t0:.0f}s)")

    # ---- Platt 학습 (2023에서만) ----
    platt = LogisticRegression()
    platt.fit(logit(p_2023).reshape(-1, 1), y_2023)
    print(f"  Platt coef={platt.coef_[0][0]:.4f}  intercept={platt.intercept_[0]:.4f}")

    # ---- Fold2: 2019-2023 -> 2024 (기존 683.8 재현 + Platt 적용) ----
    print("\n[Fold2] 2019-2023 학습 -> 2024 예측...")
    train_b = df[df["season"] <= 2023].reset_index(drop=True)
    valid_b = df[df["season"] == 2024].reset_index(drop=True)
    y_2024 = valid_b[TARGET_COL].to_numpy()
    p_2024 = train_ensemble(train_b, valid_b)
    m_2024_raw = evaluate(y_2024, p_2024)
    print(format_report("  2024 raw ensemble (기존 683.8 재현)", m_2024_raw))
    print(f"  ({time.time() - t0:.0f}s)")

    p_2024_calibrated = platt.predict_proba(logit(p_2024).reshape(-1, 1))[:, 1]
    m_2024_cal = evaluate(y_2024, p_2024_calibrated)
    print(format_report("  2024 Platt 보정 후", m_2024_cal))

    delta = m_2024_cal["bss"] - m_2024_raw["bss"]
    print(f"\nBSS 개선폭: {delta:+.6f}  (채택 기준: +{ADOPT_THRESHOLD})")
    if delta >= ADOPT_THRESHOLD:
        print("=> 채택: Platt calibration 버전을 별도 패키징")
    else:
        print("=> 폐기: 개선폭이 기준 미달, 원본(RF 0.15+HGB 0.85) 유지")

    print(f"\n총 소요 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
