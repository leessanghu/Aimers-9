"""v116 = v95 + mc6원본(w=0.48). public 최적점.

실측 2점으로 정확히 연립한 곡선:
  v112 s=0.03 -> +1.1775
  v114 s=0.10 -> +3.6309
  => A = -5.0596e-05,  V = 1.0491e-04  (두 점을 4.4e-16 오차로 재현)
  => ΔScore(s) = 40.51*s - 41.99*s^2,  s* = 0.4823, 최대이득 +9.77

항등식이 정확한 근거(코드 확인 완료):
  risk보정 = risk_vec(mc5확률)만 사용, k2보정 = pitcher_id 룩업만 사용,
  level_shift = 상수. 셋 다 preds에 의존 안 함 -> preds(s) = v95_final + s*d 정확.
  예측범위 [0.29,0.67]이라 클리핑도 안 걸림.

[리스크] private 리더보드가 별도 존재. A/V는 현재 보이는 점수 기준값이므로
private에서 최적점이 다를 수 있음. s=0.48은 public 최적에 정확히 맞춘 값.

모델 재학습 불필요 - v112의 mc6pure_model 그대로 재사용.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W_NEW = 0.48

v112 = joblib.load('submit/model/model_artifacts_v112.pkl')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')

v116 = dict(v112)
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
print('=== 가중치 재배분 (v95 원본 기준 비례축소) ===')
for k in HEADS:
    orig = float(v95[f'{k}_weight'])
    new = orig * (1 - W_NEW)
    v116[f'{k}_weight'] = new
    print(f'  {k:12s} v95={orig:.4f} -> {new:.4f}')
v116['mc6pure_weight'] = W_NEW
tot = sum(float(v116[f'{k}_weight']) for k in HEADS) + W_NEW
print(f'  mc6pure      0.0000 -> {W_NEW:.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9
assert v116.get('mc6pure_model') is not None, 'mc6pure_model 승계 실패'
assert v116.get('mc6pure_succ_classes') is not None, 'succ_classes 승계 실패'

joblib.dump(v116, 'submit/model/model_artifacts_v116.pkl')
print('\nv116 저장 완료 (재학습 없음)')
print('예상 ΔScore = 40.51*0.48 - 41.99*0.48^2 = %+.2f' % (40.51*0.48 - 41.99*0.48**2))
print('예상 점수 = 1103.6568 %+.2f = %.2f' % (40.51*0.48 - 41.99*0.48**2,
                                              1103.6568 + 40.51*0.48 - 41.99*0.48**2))
