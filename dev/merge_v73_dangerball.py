"""v73 = v66 + dangerball축(dangerous 행에서만 1-ball, cond_ball의 여집합).

codex idea55 로컬: v66에 추가시 -2.68(2시드 -2.93), count_resid 대체시 +0.70.
단 로컬은 이 계열에서 역전지표로 확인됐다(v62/63/64가 로컬 전부 음수였는데 실측
3/3 양수). 따라서 로컬 음수를 기각근거로 쓰지 않고 실측으로 판정한다.

재학습 없이 model_artifacts_dangerball.pkl의 dangerball_model만 꺼내 v66에 얹고
기존 가중치를 (1-w) 비례축소한다.
"""
import os, joblib

OUT = "../submit/model"
W = 0.08

v66 = joblib.load(os.path.join(OUT, "model_artifacts_v66.pkl"))
db = joblib.load(os.path.join(OUT, "model_artifacts_dangerball.pkl"))
assert list(v66["feature_order"]) == list(db["feature_order"]), "feature_order 불일치"

v73 = dict(v66)
for k in list(v73):
    if k.endswith("_weight") or k == "base_weight":
        v73[k] = float(v73[k]) * (1.0 - W)
v73["dangerball_model"] = db["dangerball_model"]
v73["dangerball_weight"] = W

s = sum(float(v) for k, v in v73.items() if k.endswith("_weight") or k == "base_weight")
assert abs(s - 1.0) < 1e-9, s
print("v73 가중치:", {k: round(v, 4) for k, v in sorted(v73.items())
                    if (k.endswith("_weight") or k == "base_weight") and v > 0})
print(f"합={s:.6f}")
out = os.path.join(OUT, "model_artifacts_v73.pkl")
joblib.dump(v73, out)
print(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
