"""v128 = v127(수확+0.24 + mc6aux프로브 0.01) + nn_raw 프로브(+0.01) 통합. v127은 스킵.

nn_raw 근거: fold A z=3.7 통과(오늘 2위, lt_y z=4.1 다음). 원시컨텍스트52+ID4종
  임베딩만으로 학습(가공피처 배제), 어제 NN(가공피처판)에도 직교화해서 살아남은
  신호라 '적은 피처라서 생긴 우연'이 아니라 진짜 다른 정보.
가중치 0.01: 로컬 s*=+0.088이나 부호/크기 모두 불신(오늘 xr/lty 둘 다 z>2.9인데도
  실측 부호반전). mc6aux와 동일하게 측정용 소량.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W_NNRAW = 0.01

v127 = joblib.load('submit/model/model_artifacts_v127.pkl')
v128 = dict(v127)

HEADS_ALL = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball',
             'countresid', 'future50', 'mc5', 'ingame',
             'mc6pure', 'strk', 'xgbunused', 'xgbrawid', 'lty', 'mc6aux']
scale = 1 - W_NNRAW
print('=== v128 가중치 (v127 전체 x(1-0.01) + nnraw 0.01) ===')
tot = 0.0
for k in HEADS_ALL:
    wk = f'{k}_weight'
    assert wk in v127, f'{wk} 없음!'
    old = float(v127[wk])
    new = old * scale
    v128[wk] = new
    tot += new
    print(f'  {k:12s} {old:+.4f} -> {new:+.4f}')

nr = joblib.load('dev/nnraw_production.pkl')
v128['nnraw_weight'] = W_NNRAW
v128['nnraw_models'] = nr['models']
tot += W_NNRAW
print(f'  nnraw        +0.0000 -> {W_NNRAW:+.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'합계오류 {tot}'

joblib.dump(v128, 'submit/model/model_artifacts_v128.pkl')
print('\nv128 저장 완료: submit/model/model_artifacts_v128.pkl')
print('예측: 1115.72(수확+mc6aux 예상) ± mc6aux항(±0.25) ± nnraw항(측정필요, 소량이라 최악도 작음)')
