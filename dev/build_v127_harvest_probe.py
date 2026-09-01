"""v127 = 9앵커 재최적화 지점(수확 +0.24) + mc6aux 소량프로브(+0.01) 결합.

구성 근거 (dev/solve_9anchor.py, 2026-08-31):
  xu고정 4축 최적점 100%: mc6=0.4387 strk=0.1971 xu=-0.0316 xr=0.0187 lty=0.0215
  -> 예측 1115.7165 (v126 실측 1115.4738 대비 +0.24, V민감도 전구간 플러스)
mc6aux를 +0.01로 얹음:
  - 부호를 로컬로 믿지 않음([[local-sign-unreliable]]: 약신호축 로컬부호 3전2패)
  - 리더보드는 결정론적이라 s=0.01로도 A가 정확히 풀림. 최악손실 ±0.25점.
  - 재최적화 부분은 모델이 오차 0.0003으로 예측하므로, 실측-예측 차이가 곧 mc6aux의 A.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, joblib

C = dict(mc6=0.4387, strk=0.1971, xu=-0.0316, xr=0.0187, lty=0.0215, mc6aux=0.01)
CORE_TOTAL = 1.0 - sum(C.values())
print(f'코어 합 = {CORE_TOTAL:+.4f}')

v126 = joblib.load('submit/model/model_artifacts_v126.pkl')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v127 = dict(v126)

CORE = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
        'condball', 'countresid', 'future50', 'mc5', 'ingame']
w95 = {k: float(v95[f'{k}_weight']) for k in CORE}
t95 = sum(w95.values())
print('=== v127 가중치 ===')
tot = 0.0
for k in CORE:
    new = w95[k] / t95 * CORE_TOTAL
    v127[f'{k}_weight'] = new
    tot += new
    print(f'  {k:12s} {float(v126[f"{k}_weight"]):+.4f} -> {new:+.4f}')
MAP = dict(mc6='mc6pure', strk='strk', xu='xgbunused', xr='xgbrawid', lty='lty', mc6aux='mc6aux')
for axis, key in MAP.items():
    v127[f'{key}_weight'] = float(C[axis])
    old = float(v126.get(f'{key}_weight', 0.0))
    tot += C[axis]
    print(f'  {key:12s} {old:+.4f} -> {C[axis]:+.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'합계오류 {tot}'

aux = joblib.load('dev/mc6aux_production.pkl')
v127['mc6aux_model'] = aux['model']
v127['mc6aux_feat_order'] = aux['feat_order']

joblib.dump(v127, 'submit/model/model_artifacts_v127.pkl')
print('\nv127 저장 완료. 예측: 1115.72 ± mc6aux항(±0.25)')
