"""v128 = v127(수확+0.24 + mc6aux프로브 0.01) + N1 프로브(+0.01). nn_raw 대신 N1(상위호환).

N1 근거: fold A z=2.4 통과 - nn_raw까지 직교화한 뒤에도 생존한 진짜 추가신호.
  원시컨텍스트53 + 원시비율18(스무딩 없는 as-of rate+n) + ID임베딩4종.
가중치 0.01: mc6aux와 동일 철학 - 부호/크기 불신, 측정용 소량.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W_N1 = 0.01

v127 = joblib.load('submit/model/model_artifacts_v127.pkl')
v128 = dict(v127)

HEADS_ALL = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball',
             'countresid', 'future50', 'mc5', 'ingame',
             'mc6pure', 'strk', 'xgbunused', 'xgbrawid', 'lty', 'mc6aux']
scale = 1 - W_N1
print('=== v128 가중치 (v127 전체 x(1-0.01) + N1 0.01) ===')
tot = 0.0
for k in HEADS_ALL:
    wk = f'{k}_weight'
    assert wk in v127, f'{wk} 없음!'
    old = float(v127[wk])
    new = old * scale
    v128[wk] = new
    tot += new
    print(f'  {k:12s} {old:+.4f} -> {new:+.4f}')

n1 = joblib.load('dev/n1_production.pkl')
v128['n1_weight'] = W_N1
v128['n1_models'] = n1['models']
v128['n1_raw18_feats'] = n1['raw18']
tot += W_N1
print(f'  N1           +0.0000 -> {W_N1:+.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'합계오류 {tot}'

joblib.dump(v128, 'submit/model/model_artifacts_v128.pkl')
print('\nv128(N1판) 저장 완료: submit/model/model_artifacts_v128.pkl')
