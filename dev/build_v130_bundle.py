"""v130 = v129(10앵커 전부자유 최적점, 수확 +0.40 예측) + 미실측 번들 프로브.

번들: zoneintent(-0.01, fold A z=2.6 로컬방향 음수) + ET(+0.01, z=1.8 로컬방향 양수).
합이 0이라 기존 가중치 스케일링 불필요(합계=1 유지).
linear(z=0.8)는 대조군 수준이라 제외.

기대: 수확 +0.40이 기본, 번들은 측정(최악 각 ±0.25). 살아있으면 마지막 제출에서 회수.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

v129 = joblib.load('submit/model/model_artifacts_v129.pkl')
v130 = dict(v129)

# 번들 가중치 (합 0 -> 기존 가중치 무변경)
W_ZI, W_ET = -0.01, +0.01

zi = joblib.load('dev/zoneintent_production.pkl')
v130['zoneintent_weight'] = W_ZI
v130['zoneintent_model'] = zi['model']
v130['zoneintent_succ_classes'] = zi['succ_classes']

et = joblib.load('dev/et_production_small.pkl')   # 압축판(18MB, 대형판과 상관 0.972)
v130['et_weight'] = W_ET
v130['et_model'] = et['model']

HEADS_ALL = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball',
             'countresid', 'future50', 'mc5', 'ingame',
             'mc6pure', 'strk', 'xgbunused', 'xgbrawid', 'lty', 'mc6aux', 'n1']
tot = sum(float(v130.get(f'{k}_weight', 0.0)) for k in HEADS_ALL) + W_ZI + W_ET
print('=== v130 = v129 + 번들 프로브 ===')
print(f'  zoneintent  {W_ZI:+.4f}  (z=2.6, 로컬방향 음수)')
print(f'  extratrees  {W_ET:+.4f}  (z=1.8, 로컬방향 양수)')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'합계오류 {tot}'

joblib.dump(v130, 'submit/model/model_artifacts_v130.pkl')
print('\nv130 저장 완료. 예측: 1116.03(수확) ± 번들항(각 최악 ±0.25)')
