"""v124 = 실측앵커 5개 기반 결합최적화의 보수판(중간점). 재학습 없음.

플랫 파라미터: mc6=0.4671, strk=0.1817, xu=-0.0316, 코어(10헤드) 합=0.3828.
결합 최적점(mc6 0.4398/strk 0.2605/xu -0.0333, 예측 +0.76)과 현재 v122의 중간점.
strk 축 앵커가 1개뿐이라 외삽리스크 절반으로 줄이고 이득의 75%(예측 +0.57) 회수.
모델 객체는 전부 v122에서 그대로(mc6pure/strk/xgbunused 포함).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

C_MC6, C_STRK, C_XU = 0.4671, 0.1817, -0.0316
CORE_TOTAL = 1.0 - C_MC6 - C_STRK - C_XU   # = 0.3828

v122 = joblib.load('submit/model/model_artifacts_v122.pkl')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v124 = dict(v122)

CORE = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
        'condball', 'countresid', 'future50', 'mc5', 'ingame']
w95 = {k: float(v95[f'{k}_weight']) for k in CORE}
t95 = sum(w95.values())
print(f'v95 코어합 = {t95:.6f} (1.0이어야 함)')

print('=== v124 가중치 ===')
tot = 0.0
for k in CORE:
    new = w95[k] / t95 * CORE_TOTAL
    v124[f'{k}_weight'] = new
    tot += new
    print(f'  {k:12s} {float(v122[f"{k}_weight"]):+.4f} -> {new:+.4f}')
v124['mc6pure_weight'] = C_MC6
v124['strk_weight'] = C_STRK
v124['xgbunused_weight'] = C_XU
tot += C_MC6 + C_STRK + C_XU
print(f'  mc6pure      {float(v122["mc6pure_weight"]):+.4f} -> {C_MC6:+.4f}')
print(f'  strk         {float(v122["strk_weight"]):+.4f} -> {C_STRK:+.4f}')
print(f'  xgbunused    {float(v122["xgbunused_weight"]):+.4f} -> {C_XU:+.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'가중치 합계 오류: {tot}'

# v123에서 넣었던 xgbrawid는 v122에 없음(확인)
assert 'xgbrawid_weight' not in v122 or float(v122.get('xgbrawid_weight', 0)) == 0

joblib.dump(v124, 'submit/model/model_artifacts_v124.pkl')
print('\nv124 저장 완료: submit/model/model_artifacts_v124.pkl')
print('예측 점수: 1115.57 (v122 +0.57, 결합이차곡면 중간점)')
