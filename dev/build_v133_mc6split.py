"""v133 = v131 + mc6split 프로브(+0.05). fold A z=3.5 통과 축.
가중치 +0.05 근거: 로컬 s*=+0.157이나 크기 2~4배 과대추정 관행 반영, 프로브 겸 실전투입.
(±0.05에서 로컬 A가 실제의 1/3이어도 +0.4~0.7, 부호반전시 -1.2 리스크 - z=3.5와
 '검증된 직접비교 기반 축'임을 감안한 공격적 프로브. 마지막 날이므로 상방 우선.)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W_MS = 0.05

v131 = joblib.load('submit/model/model_artifacts_v131.pkl')
v133 = dict(v131)

HEADS_ALL = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball',
             'countresid', 'future50', 'mc5', 'ingame',
             'mc6pure', 'strk', 'xgbunused', 'xgbrawid', 'lty']
scale = 1 - W_MS
print(f'=== v133 (v131 x{scale} + mc6split {W_MS}) ===')
tot = 0.0
for k in HEADS_ALL:
    wk = f'{k}_weight'
    old = float(v131[wk])
    new = old * scale
    v133[wk] = new
    tot += new
    print(f'  {k:12s} {old:+.4f} -> {new:+.4f}')

ms = joblib.load('dev/mc6split_production.pkl')
v133['mc6split_weight'] = W_MS
v133['mc6split_model_R'] = ms['model_R']
v133['mc6split_model_F'] = ms['model_F']
v133['mc6split_succ_classes'] = ms['succ_classes']
v133['mc6split_r_value'] = ms['r_value']
tot += W_MS
print(f'  mc6split     +0.0000 -> {W_MS:+.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'합계오류 {tot}'

joblib.dump(v133, 'submit/model/model_artifacts_v133.pkl')
print('\nv133 저장 완료')
