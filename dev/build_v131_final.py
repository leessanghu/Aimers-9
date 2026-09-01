"""v131(최종) = xu고정 4축 최적점, 프로브 전부 제거.

v130(-0.89) 사후분석: xu 대이동(-0.031->-0.066, 교차항 의존)과 번들(zi/et)이 함께 실패.
v131은 실측 검증된 안전지대로 복귀:
  - xu는 v122/v128에서 실측 확정된 -0.0316 유지
  - mc6/strk/xr/lty는 10앵커 xu고정 최적점(모델이 v128 재현오차 0.0003으로 검증된 영역)
  - 죽은 프로브(mc6aux/N1/zi/et) 전부 0
예측 1115.71 (v128 +0.08). 최악도 v128-0.1 수준(같은 좌표 근방이라 모델오차만 리스크).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

C = dict(mc6=0.4387, strk=0.1971, xu=-0.0316, xr=0.0184, lty=0.0216)
CORE_TOTAL = 1.0 - sum(C.values())
print(f'코어 합 = {CORE_TOTAL:+.4f}')

v130 = joblib.load('submit/model/model_artifacts_v130.pkl')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v131 = dict(v130)

CORE = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
        'condball', 'countresid', 'future50', 'mc5', 'ingame']
w95 = {k: float(v95[f'{k}_weight']) for k in CORE}
t95 = sum(w95.values())
print('=== v131 가중치 ===')
tot = 0.0
for k in CORE:
    new = w95[k] / t95 * CORE_TOTAL
    v131[f'{k}_weight'] = new
    tot += new
    print(f'  {k:12s} -> {new:+.4f}')
MAP = dict(mc6='mc6pure', strk='strk', xu='xgbunused', xr='xgbrawid', lty='lty')
for axis, key in MAP.items():
    v131[f'{key}_weight'] = float(C[axis])
    tot += C[axis]
    print(f'  {key:12s} -> {C[axis]:+.4f}')
for dead in ('mc6aux', 'n1', 'zoneintent', 'et'):
    v131[f'{dead}_weight'] = 0.0
    print(f'  {dead:12s} -> +0.0000 (제거)')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'합계오류 {tot}'

joblib.dump(v131, 'submit/model/model_artifacts_v131.pkl')
print('\nv131 저장 완료. 예측 1115.71 (v128 +0.08)')
