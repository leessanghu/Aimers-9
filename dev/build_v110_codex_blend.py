"""v110 = v95 + codex v20_905 선형블렌드(w=0.15).

v95 헤드 가중치는 그대로 두고(내부 합=1 유지), 최종 예측 단계에서
  final = (1-0.15)*v95_preds + 0.15*codex_preds
로 결합한다. codex는 완전히 독립 파이프라인이라 헤드로 흡수하지 않고 바깥에서 섞는다.

[주의] codex 모델은 2019-2024 전체 in-sample이라 fold A/C 둘 다 오염 -> 로컬검증 불가.
  fold A: 우리923.8 vs codex1651.6 (s*=+1.33)
  fold C: 우리2584.7 vs codex3065.4 (s*=+1.13)
  s*>1 = 자기 학습데이터로 평가할 때 나오는 패턴. 검증 미통과 실험적 제출.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W = 0.15
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v110 = dict(v95)
v110['codex_weight'] = W

HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
tot = sum(float(v110[f'{k}_weight']) for k in HEADS)
print(f'v95 내부 헤드 가중치 합 = {tot:.6f} (그대로 유지)')
print(f'codex_weight = {W}  ->  final = {1-W:.2f}*v95 + {W:.2f}*codex')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v110, 'submit/model/model_artifacts_v110.pkl')
print('v110 저장 완료')
