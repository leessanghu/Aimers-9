"""4모델(RF/HGB/LGBM-A/LGBM-D) 앙상블 아티팩트 조립.
가중치: RF+HGB 55%(내부 0.15/0.85 유지) + LGBM-A 35% + LGBM-D 10%
(CatBoost는 아직 학습 중이라 이번 라운드는 제외, 준비되면 재조립)
"""
import joblib

a1 = joblib.load("../submit/model/model_artifacts.pkl")
a2 = joblib.load("../submit/model/model_artifacts_v2.pkl")

RF_HGB_TOTAL = 0.55
W_A = 0.35
W_D = 0.10

artifacts = {
    "rf": a1["rf"], "hgb": a1["hgb"], "stats_rfhgb": a1["stats"],
    "w_rf": 0.15 * RF_HGB_TOTAL, "w_hgb": 0.85 * RF_HGB_TOTAL,
    "model_a": a2["model_a"], "stats_a": a2["stats_a"], "w_a": W_A,
    "model_d": a2["model_d"], "stats_d": a2["stats_d"], "drop_cols_d": a2["drop_cols_d"], "w_d": W_D,
}
total = artifacts["w_rf"] + artifacts["w_hgb"] + artifacts["w_a"] + artifacts["w_d"]
print("weights:", {k: v for k, v in artifacts.items() if k.startswith("w_")}, " sum=", total)
assert abs(total - 1.0) < 1e-9

joblib.dump(artifacts, "../submit/model/model_artifacts_v3.pkl")
print("저장 완료: submit/model/model_artifacts_v3.pkl")
