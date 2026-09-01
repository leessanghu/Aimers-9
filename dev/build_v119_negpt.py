"""v119 = v95 + mc6(0.478) + strk(0.256) + pitchtype(-0.14).

3축 결합 최적화 결과(민감도 검증 완료, dev/sensitivity_pt_negative.py):
  A1=-5.0596e-05(mc6, 실측2점 확정) A2=-2.9235e-05(strk, 실측1점) A3=-3.19e-06(pt, 실측1점)
  s*=(0.478, 0.256, -0.140), 예상Δ=+12.50, 예상점수=1116.15
  민감도: V13/V23/V33을 ±50~100%(2000회 랜덤스윕) 흔려도 s3*<0 100%, 예상점수 1115.4~1122.0
  -> 하방 거의 없음(최악도 "pt 제외"보다 나음), 부호 안정적.

재학습 불필요 - v118의 mc6pure_model/strk_model/pitchtype_model 전부 재사용, 가중치만 변경.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

S_MC6, S_STRK, S_PT = 0.478, 0.256, -0.140

v118 = joblib.load('submit/model/model_artifacts_v118.pkl')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')

v119 = dict(v118)
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
rest = 1.0 - S_MC6 - S_STRK - S_PT
print(f'=== 가중치 (v95 원본 x {rest:.4f}) ===')
for k in HEADS:
    orig = float(v95[f'{k}_weight'])
    v119[f'{k}_weight'] = orig * rest
    print(f'  {k:12s} v95={orig:.4f} -> {orig*rest:.4f}')
v119['mc6pure_weight'] = S_MC6
v119['strk_weight'] = S_STRK
v119['pitchtype_weight'] = S_PT
tot = sum(float(v119[f'{k}_weight']) for k in HEADS) + S_MC6 + S_STRK + S_PT
print(f'  mc6pure      -> {S_MC6:.4f}')
print(f'  strk         -> {S_STRK:.4f}')
print(f'  pitchtype    -> {S_PT:+.4f}  (음수)')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9
for k in ('mc6pure_model', 'strk_model', 'pitchtype_model'):
    assert v119.get(k) is not None, f'{k} 누락'

joblib.dump(v119, 'submit/model/model_artifacts_v119.pkl')
print('\nv119 저장 완료 (재학습 없음)')
print('예상점수 = 1116.15')
