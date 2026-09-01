"""v121 = v117 구조 + 헤드 재배분(t=0.12). [버그수정판]
overlap4(midother/condball/countresid/future50)의 12%를 indep3(base/hurdle/ordinal)로 이전.
재학습 불필요 - v117의 모든 모델 그대로, 8헤드 '내부'에서만 질량 이동(외부 재정규화 없음).

이전 버전 버그: v95 8헤드만으로 재정규화한 비율에 rest를 곱해서 mc5/ingame 몫을
중복 배분했음(합계 1.0916으로 깨짐). 이번엔 v117의 실제 8헤드 가중치를 그대로 기준으로
삼아 그 풀 안에서만 이동 -> mc5/ingame/mc6/strk는 완전히 안 건드림, 합계 자동으로 1 유지.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

T = 0.12
OVERLAP = ['midother', 'condball', 'countresid', 'future50']
INDEP = ['base', 'hurdle', 'ordinal']
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']

v117 = joblib.load('submit/model/model_artifacts_v117.pkl')

W0 = {k: float(v117[f'{k}_weight']) for k in HEADS8}   # v117의 실제 가중치(그대로)
overlap_sum = sum(W0[k] for k in OVERLAP)
indep_sum = sum(W0[k] for k in INDEP)
move = overlap_sum * T

v121 = dict(v117)
print('=== 헤드 재배분 (t=0.12, 8헤드 내부에서만 이동) ===')
for k in HEADS8:
    if k in OVERLAP:
        new = W0[k] * (1 - T)
    elif k in INDEP:
        new = W0[k] + move * (W0[k] / indep_sum)
    else:  # multires: 변경없음
        new = W0[k]
    v121[f'{k}_weight'] = new
    print(f'  {k:12s} v117={W0[k]:.4f} -> {new:.4f}')

ALL = HEADS8 + ['mc5', 'ingame', 'mc6pure', 'strk']
tot = sum(float(v121.get(f'{k}_weight', 0.0)) for k in ALL)
print(f'  mc5      {v121["mc5_weight"]:.4f} (변경없음)')
print(f'  ingame   {v121["ingame_weight"]:.4f} (변경없음)')
print(f'  mc6pure  {v121["mc6pure_weight"]:.4f} (변경없음)')
print(f'  strk     {v121["strk_weight"]:.4f} (변경없음)')
print(f'  진짜 합계(전체) = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'합계 불일치: {tot}'

joblib.dump(v121, 'submit/model/model_artifacts_v121.pkl')
print('\nv121 저장 완료 (재학습 없음, v117 모델 전부 재사용)')
