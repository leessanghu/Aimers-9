"""v42 = v35(base+hurdle) + v40의 multires_model + v41의 ordinal_stage1/2/3, 재학습 없이 병합.
가중치는 idea16 심플렉스 그리드서치 최적조합: base=0.30 hur=0.40 mr=0.10 or=0.20.
"""
import os

import joblib

OUT_DIR = "../submit/model"

v40 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v40.pkl"))
v41 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v41.pkl"))

common = dict(v40)  # v35 base + hurdle + multires_model 이미 포함
common["ordinal_stage1"] = v41["ordinal_stage1"]
common["ordinal_stage2"] = v41["ordinal_stage2"]
common["ordinal_stage3"] = v41["ordinal_stage3"]

W_BASE, W_HUR, W_MR, W_OR = 0.30, 0.40, 0.10, 0.20
common["hurdle_weight"] = W_HUR
common["multires_weight"] = W_MR
common["ordinal_weight"] = W_OR
common["mix_weight"] = 0.0
common["denoise_weight"] = 0.0
common["multi_weight"] = 0.0
common["base_weight"] = W_BASE

assert abs(W_BASE + W_HUR + W_MR + W_OR - 1.0) < 1e-9

out = os.path.join(OUT_DIR, "model_artifacts_v42.pkl")
joblib.dump(common, out)
print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
print(f"weights: base={W_BASE} hurdle={W_HUR} multires={W_MR} ordinal={W_OR}")
