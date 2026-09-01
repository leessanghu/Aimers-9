"""v76 = v66 + 5-class softmax(w=0.15). v74(w=0.05) 실측 +4.22를 곡률 외삽해
실측 최적 w*=0.148로 추정 -> 0.15 채택. 재학습 없이 v74의 mc5_model 재사용.
"""
import os, joblib
OUT = "../submit/model"
W = 0.15
v66 = joblib.load(os.path.join(OUT, "model_artifacts_v66.pkl"))
v74 = joblib.load(os.path.join(OUT, "model_artifacts_v74.pkl"))
v76 = dict(v66)
for k in list(v76):
    if k.endswith("_weight") or k == "base_weight":
        v76[k] = float(v76[k]) * (1.0 - W)
v76["mc5_model"] = v74["mc5_model"]
v76["mc5_succ"] = v74["mc5_succ"]
v76["mc5_weight"] = W
s = sum(float(v) for k, v in v76.items() if k.endswith("_weight") or k == "base_weight")
assert abs(s - 1.0) < 1e-9, s
print("v76 가중치:", {k: round(v,4) for k,v in sorted(v76.items()) if (k.endswith("_weight") or k=="base_weight") and v>0})
print(f"합={s:.6f}")
out = os.path.join(OUT, "model_artifacts_v76.pkl")
joblib.dump(v76, out)
print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
