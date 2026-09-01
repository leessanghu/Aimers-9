"""mc6를 최적점(s=0.48)에 넣은 지금, 추가로 벌 수 있는 축이 뭔지 판정.

핵심: 두 축을 같이 넣으면
  BS = BS0 + 2s1*A1 + 2s2*A2 + s1^2*V11 + 2*s1*s2*V12 + s2^2*V22
s1이 이미 고정됐을 때 축2의 유효신호는 (A2 + s1*V12) 이다.
=> V12(축간 공분산)가 크면 이미 mc6가 먹은 방향이라 추가이득 없음.

V12는 y가 안 들어가는 '순수 예측공간' 양이라 로컬로 추정 가능(A와 달리).
fold A에서 각 후보의 d벡터 상관을 재서 mc6와 직교한 축을 찾는다.
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

va = season == 2024
yv = y_all[va]
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
resid = yv - blend
E_resid2 = float(np.mean(resid ** 2))

CANDS = {
    'mc6_pure':      'dev/cache_mc6head_A.npy',
    'mc6h_wild':     'dev/cache_mc6h_headA_wild_A.npy',
    'mc6h_ball':     'dev/cache_mc6h_headB_ball_A.npy',
    'mc6h_strike':   'dev/cache_mc6h_headC_strike_A.npy',
    'strk_linear':   'dev/cache_strk_strk_linear_A.npy',
    'strk_tail5':    'dev/cache_strk_strk_tail5_A.npy',
    'seqC_prevball': 'dev/cache_seq_seqC_prev_ball_A.npy',
    'seqA_prev_y':   'dev/cache_seq_seqA_prev_y_A.npy',
    'pitchtype':     'dev/cache_pitchtypehead_A.npy',
    'persona':       'dev/cache_persona_A.npy',
    'xgb_rawid':     'dev/cache_xgbrawid_A.npy',
}
D, meta_rows = {}, []
for nm, path in CANDS.items():
    if not os.path.exists(path):
        print(f'  [skip] {nm}: 캐시없음')
        continue
    p = np.load(path)
    d = p - blend
    d = d - d.mean()
    D[nm] = d
    A = float(np.mean(d * (blend - yv)))
    V = float(np.mean(d ** 2))
    rho = -A / np.sqrt(V * E_resid2)
    meta_rows.append((nm, A, V, rho, K * A ** 2 / V))

print('=== fold A 기준 각 후보 (참고: 로컬은 실측과 크게 다를 수 있음) ===')
print(f'{"후보":<16}{"A":>12}{"V":>11}{"rho":>10}{"로컬최대이득":>13}')
for nm, A, V, rho, g in meta_rows:
    print(f'{nm:<16}{A:>+12.3e}{V:>11.3e}{rho:>+10.5f}{g:>13.2f}')

names = list(D.keys())
M = np.column_stack([D[n] for n in names])
C = np.corrcoef(M.T)
print(f'\n=== d벡터 상관행렬 (mc6_pure와의 상관이 핵심) ===')
print(f'{"후보":<16}' + ''.join(f'{n[:9]:>10}' for n in names))
for i, n in enumerate(names):
    print(f'{n:<16}' + ''.join(f'{C[i, j]:>10.3f}' for j in range(len(names))))

i_mc6 = names.index('mc6_pure')
print(f'\n=== mc6_pure(s=0.48 적용중)와의 상관 순 — 낮을수록 신규축 ===')
order = np.argsort(np.abs(C[i_mc6]))
for j in order:
    if names[j] == 'mc6_pure':
        continue
    corr = C[i_mc6, j]
    print(f'  {names[j]:<16} corr={corr:+.3f}   '
          f'{"독립적(유망)" if abs(corr) < 0.5 else "중복(mc6가 이미 먹음)" if abs(corr) > 0.8 else "부분중복"}')

# mc6 실측값으로 스케일 보정한 '유효신호' 추정
A1_real, V11_real, s1 = -5.0596e-05, 1.0491e-04, 0.48
scale = np.sqrt(V11_real / D['mc6_pure'].var())   # 로컬 V -> 실측 V 스케일
print(f'\n=== mc6가 s={s1}로 이미 들어간 상태에서 각 축의 잔여 유효신호 추정 ===')
print(f'  (로컬 A는 신뢰 못 하므로 V12 항만 실측스케일로 보정해서 "얼마나 깎이는지"만 표시)')
print(f'{"후보":<16}{"V12(보정)":>13}{"s1*V12":>12}{"해석":>28}')
for j, n in enumerate(names):
    if n == 'mc6_pure':
        continue
    V12 = float(np.mean(D['mc6_pure'] * D[n])) * scale ** 2
    print(f'{n:<16}{V12:>+13.3e}{s1*V12:>+12.3e}'
          f'{"  mc6가 이미 상당부분 흡수" if abs(s1*V12) > 3e-5 else "  대부분 신규신호":>28}')
