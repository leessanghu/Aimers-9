"""최종 제출 모델 학습: 2019~2024 전체 train.csv로 FeatureBuilder + RF/HGB를 학습하고
submit/model/model_artifacts.pkl 로 저장한다 (script.py가 그대로 로드해서 추론).

최종 채택안: RF 0.15 + HGB 0.85 가중 앙상블, smoothed-only 피처(raw rate 미포함).
"""

import os
import time

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from features import FeatureBuilder, TARGET_COL

DATA_PATH = "../data/train.csv"
OUT_DIR = "../submit/model"
SEED = 42

RF_PARAMS = dict(n_estimators=300, max_depth=10, min_samples_leaf=200, n_jobs=-1, random_state=SEED)
HGB_PARAMS = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
                   early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
W_RF = 0.15

_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")


def strip_rng(obj, seen=None, depth=0):
    """fit된 모델 안에 남아있는 numpy RNG(Generator/BitGenerator/RandomState) 인스턴스를
    전부 None으로 지운다. 학습 전용 상태라 predict/predict_proba엔 영향 없음.
    numpy 메이저 버전 간 pickle 비호환(2.x로 저장 -> 1.x에서 로드 실패)을 피하기 위함."""
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG_TYPES:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strip_rng(v, seen, depth + 1)


def main():
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    print(f"전체 train={len(df):,}  season {df['season'].min()}~{df['season'].max()}")
    y = df[TARGET_COL].to_numpy()

    fb = FeatureBuilder(seed=SEED, include_raw_rates=False).fit(df)
    X = fb.transform_train_oof(df)
    print(f"피처 수={X.shape[1]}  ({time.time() - t0:.0f}s)")

    rf = RandomForestClassifier(**RF_PARAMS).fit(X, y)
    print(f"RF 완료 ({time.time() - t0:.0f}s)")

    hgb = HistGradientBoostingClassifier(**HGB_PARAMS).fit(X, y)
    print(f"HGB 완료 ({time.time() - t0:.0f}s)")

    # numpy>=2.0 로컬 환경에서 학습하면 HGB가 내부적으로 만드는
    # _feature_subsample_rng(numpy.random.Generator)가 pickle에 같이 저장되는데,
    # 평가 서버(numpy==1.26.4, numpy<2.0)에서 이걸 못 읽어 "PCG64 is not a known
    # BitGenerator module" 로 로드 자체가 실패한다. predict_proba엔 안 쓰이는
    # 학습 전용 속성이라 안전하게 지운다.
    strip_rng(rf)
    strip_rng(hgb)

    os.makedirs(OUT_DIR, exist_ok=True)
    artifacts = {
        "stats": fb.export_stats(),
        "rf": rf,
        "hgb": hgb,
        "w_rf": W_RF,
        "w_hgb": round(1 - W_RF, 2),
    }
    out_path = os.path.join(OUT_DIR, "model_artifacts.pkl")
    joblib.dump(artifacts, out_path)
    size_mb = os.path.getsize(out_path) / 1e6
    print(f"저장 완료: {out_path} ({size_mb:.1f}MB)  총 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
