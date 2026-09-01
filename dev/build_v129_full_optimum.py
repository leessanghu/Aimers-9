"""v129 = 10앵커 전부자유 최적점. 죽은 프로브(mc6aux/N1) 제거(-0.08 회수).

좌표(solve_10anchor): mc6=0.4388 strk=0.2007 xu=-0.0664 xr=0.0458 lty=0.0195
예측 1116.0332 (v128 대비 +0.40). 리스크: xu 이동(-0.031->-0.066)은 교차항 의존.
실패시 다음 제출에서 xu고정 최적점(1115.71)으로 후퇴 가능 - 리스크 사다리 구조.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

C = dict(mc6=0.4388, strk=0.2007, xu=-0.0664, xr=0.0458, lty=0.0195)
CORE_TOTAL = 1.0 - sum(C.values())
print(f'코어 합 = {CORE_TOTAL:+.4f}')

v128 = joblib.load('submit/model/model_artifacts_v128.pkl')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v129 = dict(v128)

CORE = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
        'condball', 'countresid', 'future50', 'mc5', 'ingame']
w95 = {k: float(v95[f'{k}_weight']) for k in CORE}
t95 = sum(w95.values())
print('=== v129 가중치 ===')
tot = 0.0
for k in CORE:
    new = w95[k] / t95 * CORE_TOTAL
    v129[f'{k}_weight'] = new
    tot += new
    print(f'  {k:12s} {float(v128[f"{k}_weight"]):+.4f} -> {new:+.4f}')
MAP = dict(mc6='mc6pure', strk='strk', xu='xgbunused', xr='xgbrawid', lty='lty')
for axis, key in MAP.items():
    v129[f'{key}_weight'] = float(C[axis])
    tot += C[axis]
    print(f'  {key:12s} {float(v128[f"{key}_weight"]):+.4f} -> {C[axis]:+.4f}')
# 죽은 프로브 제거
v129['mc6aux_weight'] = 0.0
v129['n1_weight'] = 0.0
print(f'  mc6aux       {float(v128["mc6aux_weight"]):+.4f} -> +0.0000 (실측 사망)')
print(f'  N1           {float(v128["n1_weight"]):+.4f} -> +0.0000 (실측 사망)')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'합계오류 {tot}'

joblib.dump(v129, 'submit/model/model_artifacts_v129.pkl')
print('\nv129 저장 완료. 예측 1116.03 (v128 +0.40)')
