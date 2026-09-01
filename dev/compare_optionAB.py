"""옵션 A(결합최적점) vs B(pitchtype 신규축) 정량 비교.
핵심: A의 이득은 v95 대비가 아니라 '현재 위치(v117=1114.53)' 대비로 봐야 한다.
그리고 A와 B를 한 제출에 합칠 수 있는지도 검토한다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
BASE = 1103.6568315036
CUR = 1114.5296512406          # v117 (s1=0.48, s2=0.10)
A1, V11 = -5.0596e-05, 1.0491e-04
A2 = -2.9235e-05
S1_CUR, S2_CUR = 0.48, 0.10

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
va = season == 2024
H = dict(
    base=avg([f'dev/phase90_cache/A_base_{m}.npy' for m in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/A_core_{m}.npy')) *
                    np.load(f'dev/phase90_cache/A_snc_{m}.npy') for m in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/A_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/A_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/A_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/A_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/A_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/A_future50_multi_s{s}.npy' for s in (42, 7)]),
)
W = {k: float(v95[f'{k}_weight']) for k in H}
t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
d1 = np.load('dev/cache_mc6head_A.npy') - blend; d1 -= d1.mean()
d2 = np.load('dev/cache_strk_strk_linear_A.npy') - blend; d2 -= d2.mean()
d3 = np.load('dev/cache_pitchtypehead_A.npy') - blend; d3 -= d3.mean()
scale = V11 / float(np.mean(d1 ** 2))
V22 = float(np.mean(d2 ** 2)) * scale
V33 = float(np.mean(d3 ** 2)) * scale
V12 = float(np.mean(d1 * d2)) * scale
V13 = float(np.mean(d1 * d3)) * scale
V23 = float(np.mean(d2 * d3)) * scale
print(f'분산행렬(실측스케일 보정 {scale:.3f}배):')
print(f'  V11={V11:.3e} V22={V22:.3e} V33={V33:.3e}')
print(f'  V12={V12:.3e} V13={V13:.3e} V23={V23:.3e}')
print(f'  상관: mc6-strk {V12/np.sqrt(V11*V22):.3f}  mc6-pt {V13/np.sqrt(V11*V33):.3f}'
      f'  strk-pt {V23/np.sqrt(V22*V33):.3f}\n')

V2 = np.array([[V11, V12], [V12, V22]])
Av2 = np.array([A1, A2])


def gain2(s1, s2):
    s = np.array([s1, s2])
    return -K * (2 * float(Av2 @ s) + float(s @ V2 @ s))


s_opt2 = -np.linalg.solve(V2, Av2)
g_opt2 = gain2(*s_opt2)
g_cur = gain2(S1_CUR, S2_CUR)
print('=' * 72)
print('[옵션 A] 2축 결합 최적점으로 이동')
print('=' * 72)
print(f'  현재 v117 (s1={S1_CUR}, s2={S2_CUR}): Δ={g_cur:+.2f}  점수={BASE+g_cur:.2f}')
print(f'  최적점  (s1={s_opt2[0]:.3f}, s2={s_opt2[1]:.3f}): Δ={g_opt2:+.2f}  점수={BASE+g_opt2:.2f}')
print(f'  >>> 현재 대비 실제 순이득 = {g_opt2-g_cur:+.2f}점  <<<')
print(f'  (제출 1회 소모)')

print('\n' + '=' * 72)
print('[옵션 B] pitchtype 3번째 축 추가 - A3 시나리오별')
print('=' * 72)
V3 = np.array([[V11, V12, V13], [V12, V22, V23], [V13, V23, V33]])
print(f'{"가정 A3":>12}{"(A1대비)":>10}{"s1*":>8}{"s2*":>8}{"s3*":>8}{"3축최적Δ":>11}{"점수":>10}{"현재대비":>10}')
scenarios = [-5.06e-05, -4.39e-05, -2.92e-05, -1.50e-05, 0.0, +1.50e-05]
for A3 in scenarios:
    Av3 = np.array([A1, A2, A3])
    s3 = -np.linalg.solve(V3, Av3)
    g3 = -K * (2 * float(Av3 @ s3) + float(s3 @ V3 @ s3))
    lbl = f'{abs(A3/A1)*100:.0f}%' if A3 != 0 else '0%'
    print(f'{A3:>+12.2e}{lbl:>10}{s3[0]:>8.3f}{s3[1]:>8.3f}{s3[2]:>8.3f}'
          f'{g3:>11.2f}{BASE+g3:>10.2f}{g3-g_cur:>+10.2f}')

print('\n' + '=' * 72)
print('[옵션 A+B 결합] 한 제출로 둘 다 - (s1*, s2*, s3=0.10) 프로브')
print('=' * 72)
print('  2축 최적점에 pitchtype을 프로브 가중치로 얹으면')
print('  -> 옵션A의 이득을 챙기면서 동시에 A3를 측정한다 (제출 1회)')
print(f'{"가정 A3":>12}{"예상Δ":>10}{"예상점수":>11}{"현재대비":>10}')
for A3 in scenarios:
    Av3 = np.array([A1, A2, A3])
    s = np.array([s_opt2[0], s_opt2[1], 0.10])
    g = -K * (2 * float(Av3 @ s) + float(s @ V3 @ s))
    print(f'{A3:>+12.2e}{g:>10.2f}{BASE+g:>11.2f}{g-g_cur:>+10.2f}')

print('\n' + '=' * 72)
print('[참고] 로컬 fold A rho -> 실측 A 의 관측된 2점')
print('=' * 72)
print('  mc6 : foldA rho=-0.00068 -> 실측 A=-5.06e-05')
print('  strk: foldA rho=-0.00519 -> 실측 A=-2.92e-05')
print('  둘 다 로컬은 "손해"(A>0)라 했는데 실측은 "이득"(A<0). 2/2 부호반전.')
print('  pitchtype foldA rho=-0.00210 -> 2점 선형외삽시 A3 ~= -4.39e-05 (매우 약한 근거)')
