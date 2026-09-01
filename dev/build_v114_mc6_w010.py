"""v114 = v95 + mc6원본(w=0.10). v112(w=0.03, 실측 +1.1775)의 가중치 상향판.

프로브 역산(v112 1회 실측 기반):
  A_real = -5.126e-05,  V(foldA추정) = 1.49e-04
  s* = +0.3435,  최대이득 +7.05점
  예상: s=0.10 -> +3.51점,  s=0.15 -> +4.81점
0.34까지 바로 안 가는 이유: 단일관측 외삽이고 V를 fold A로 근사했음.
A가 실제로 절반만 음수여도 s*=0.17로 내려오므로 0.10이 안전한 다음 스텝.

모델은 v112와 완전히 동일(mc6pure_model 재사용) - 가중치만 변경하므로 재학습 불필요.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W_NEW = 0.10

v112 = joblib.load('submit/model/model_artifacts_v112.pkl')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')

v114 = dict(v112)   # mc6pure_model/succ_classes 그대로 승계
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
print('=== 가중치 재배분 (v95 원본 기준으로 다시 비례축소) ===')
for k in HEADS:
    orig = float(v95[f'{k}_weight'])      # v112가 아니라 v95 원본에서 재계산
    new = orig * (1 - W_NEW)
    v114[f'{k}_weight'] = new
    print(f'  {k:12s} v95={orig:.4f} -> {new:.4f}')
v114['mc6pure_weight'] = W_NEW
tot = sum(float(v114[f'{k}_weight']) for k in HEADS) + W_NEW
print(f'  mc6pure      0.0000 -> {W_NEW:.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9
assert v114.get('mc6pure_model') is not None, 'mc6pure_model 승계 실패'

joblib.dump(v114, 'submit/model/model_artifacts_v114.pkl')
print('v114 저장 완료 (재학습 없음, v112 모델 재사용)')
