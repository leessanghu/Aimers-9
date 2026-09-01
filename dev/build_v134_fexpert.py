"""v134 = v131 + F전문가 가산 프로브(+0.01, 블라인드 - fold A 전체축 스크린 없이).
0.01이라 최악 ~±0.5, v131의 +0.08 쿠션. 실측으로 F전문가 축 A를 측정."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W_FX = 0.01

v131 = joblib.load('submit/model/model_artifacts_v131.pkl')
v134 = dict(v131)

HEADS_ALL = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball',
             'countresid', 'future50', 'mc5', 'ingame',
             'mc6pure', 'strk', 'xgbunused', 'xgbrawid', 'lty']
scale = 1 - W_FX
tot = 0.0
for k in HEADS_ALL:
    wk = f'{k}_weight'
    new = float(v131[wk]) * scale
    v134[wk] = new
    tot += new

fx = joblib.load('dev/fexpert_prod.pkl')
v134['fexadd_weight'] = W_FX
v134['fexadd_model'] = fx['model']
v134['fexadd_succ_classes'] = fx['succ_classes']
tot += W_FX
print(f'합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v134, 'submit/model/model_artifacts_v134.pkl')
print('v134 저장 완료 (v131 x0.99 + fexpert +0.01)')
