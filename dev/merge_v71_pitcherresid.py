"""v71 = v66(현재 최고, 실측 1085.21) + pitcherresid축(y - 투수시즌LOO실력).
재학습 없이 v70에서 학습된 pitcherresid_model 객체만 꺼내 v66 위에 얹는다.
기존 v66의 모든 가중치(base/hurdle/multires/ordinal/midother/condball/countresid/
future50)를 (1-new_weight) 비율로 비례축소한다.
"""
import os

import joblib

OUT = "../submit/model"
NEW_WEIGHT = 0.10


def rescale_weights(artifact: dict, factor: float) -> None:
    for key in list(artifact):
        if key.endswith("_weight") or key == "base_weight":
            artifact[key] = float(artifact[key]) * factor


v66 = joblib.load(os.path.join(OUT, "model_artifacts_v66.pkl"))
v70 = joblib.load(os.path.join(OUT, "model_artifacts_v70.pkl"))
assert list(v66["feature_order"]) == list(v70["feature_order"]), "feature_order 불일치"

v71 = dict(v66)
rescale_weights(v71, 1.0 - NEW_WEIGHT)
v71["pitcherresid_model"] = v70["pitcherresid_model"]
v71["pitcherresid_weight"] = NEW_WEIGHT

weight_sum = sum(float(v) for k, v in v71.items() if k.endswith("_weight") or k == "base_weight")
assert abs(weight_sum - 1.0) < 1e-9, weight_sum
print(f"v71 weights: {[(k, round(v,4)) for k, v in v71.items() if k.endswith('_weight') or k == 'base_weight']}")
print(f"weight_sum={weight_sum:.6f}")

out = os.path.join(OUT, "model_artifacts_v71.pkl")
joblib.dump(v71, out)
print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
