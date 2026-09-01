"""v80 = v79 재구성: 중복덩어리 축소 + 독립멤버 강화. 재학습 없음(가중치만 변경).

근거(fold A 2024 상관 실측):
  multires/midother/condball/countresid/future50 5개가 서로 0.977~0.990
  (내부 평균상관 0.9811)인데 총 가중치의 39.4%를 차지한다. 전체 가중 N_eff=0.147.
  반면 mc5는 나머지 전부와 0.736~0.799로 가장 독립적이다.

우리는 80회 제출하며 매번 "새 축 추가 -> 기존 전체 비례축소"를 반복했고, 그 결과
이 덩어리가 통째로 유지되면서 독립 멤버가 못 컸다. 5개 중 midother 하나만 남기고
(로컬상 B:완전제거=-4.31로 덩어리 자체는 유효, A:하나만남김=+1.89로 최적)
남은 가중치를 base/hurdle/ordinal/mc5에 재배분한다.

mc5는 실측 곡선상 11-class 최적 w=0.176이므로 그 근처로 올린다.
ingame은 실측 곡선상 최적 w=0.069이므로 0.07로 소폭 조정.
"""
import os, joblib

OUT = "../submit/model"
v79 = joblib.load(os.path.join(OUT, "model_artifacts_v79.pkl"))

NEW_W = {
    "base_weight":       0.21,
    "hurdle_weight":     0.25,
    "ordinal_weight":    0.13,
    "midother_weight":   0.15,
    "mc5_weight":        0.19,
    "ingame_weight":     0.07,
}
DROP = ["multires_weight", "condball_weight", "countresid_weight", "future50_weight"]

v80 = dict(v79)
for k in DROP:
    v80[k] = 0.0
for k, v in NEW_W.items():
    v80[k] = v

s = sum(float(v) for k, v in v80.items() if k.endswith("_weight") or k == "base_weight")
assert abs(s - 1.0) < 1e-9, s
print("v80 가중치:")
for k, v in sorted(v80.items()):
    if (k.endswith("_weight") or k == "base_weight") and v > 0:
        print(f"  {k:22s} {v:.4f}")
print(f"  합={s:.6f}")
print(f"\n제거된 멤버(가중치 0): {DROP}")
print("  (모델 객체는 아티팩트에 남지만 weight=0이라 script.py가 건너뜀)")

out = os.path.join(OUT, "model_artifacts_v80.pkl")
joblib.dump(v80, out)
print(f"\n저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
