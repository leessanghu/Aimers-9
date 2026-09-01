"""v125 = v124(실측 1115.1606262971) + lt_y 헤드(음수가중치 w=-0.03) 프로브.

lt_y: linear_tree LGBM(조각별 선형 함수공간), binary y.
근거(2026-08-31 fold A): 직교화(vs d_mc6, d_xu) 후 rho=-0.00636, 순열대조군 z=4.1
  - 오늘 스크리닝 통과 후보 중 최강. 방향 음수(계단모델들의 공통편향을 과장 -> 빼기).
가중치 -0.03: 로컬 s*=-0.15이나 로컬 크기 과대추정(2~15배) 감안 보수적 선택.
  로컬 A가 4배 부풀려진 경우에도 기대 +0.24, 부호반전시 최악 -0.55 수준.
기준조합을 v124 그대로 두는 이유: v124는 정확한 실측 앵커(1115.1606)라서
  이 프로브 하나로 lt_y축 A를 오차 없이 역산 가능.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W_NEW = -0.03

v124 = joblib.load('submit/model/model_artifacts_v124.pkl')
v125 = dict(v124)

HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame',
         'mc6pure', 'strk', 'xgbunused']
scale = 1 - W_NEW   # 1.03
print('=== v125 가중치 (v124 전체 x1.03 + lty -0.03) ===')
tot = 0.0
for k in HEADS:
    wk = f'{k}_weight'
    assert wk in v124, f'{wk} 없음!'
    old = float(v124[wk])
    new = old * scale
    v125[wk] = new
    tot += new
    print(f'  {k:12s} {old:+.4f} -> {new:+.4f}')

lty = joblib.load('dev/lty_production.pkl')
v125['lty_weight'] = W_NEW
v125['lty_model'] = lty['model']
v125['lty_feat_order'] = lty['feat_order']
tot += W_NEW
print(f'  lty          +0.0000 -> {W_NEW:+.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'가중치 합계 오류: {tot}'

joblib.dump(v125, 'submit/model/model_artifacts_v125.pkl')
print('\nv125 저장 완료: submit/model/model_artifacts_v125.pkl')
