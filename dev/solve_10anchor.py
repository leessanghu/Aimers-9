"""10앵커(v128 포함) 전체 재적합 + v128 편차의 원인 분해.

v128 플랫좌표(기존 5축) = v127좌표 x 0.99:
  mc6=0.4343, strk=0.1951, xu=-0.0313, xr=0.0185, lty=0.0213 (+aux 0.0099, N1 0.0100)
기존 5축 모델로 이 좌표의 예측치(S_base)를 구하고,
실측과의 차이 = (mc6aux+N1 번들의 실효 A) + 모델오차.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
H = dict(
    base=avg([f'dev/phase90_cache/A_base_{q}.npy' for q in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/A_core_{q}.npy')) *
                    np.load(f'dev/phase90_cache/A_snc_{q}.npy') for q in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/A_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/A_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/A_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/A_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/A_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/A_future50_multi_s{s}.npy' for s in (42, 7)]),
)
W0 = {k: float(v95a[f'{k}_weight']) for k in HEADS8}
t_ = sum(W0.values()); W0 = {k: v / t_ for k, v in W0.items()}
core = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
NAMES = ['mc6', 'strk', 'xu', 'xr', 'lty']
P = [np.load('dev/cache_mc6head_A.npy'),
     np.load('dev/cache_strk_strk_linear_A.npy'),
     np.load('dev/cache_xgbunused_A.npy'),
     np.load('dev/cache_xgbrawid_A.npy'),
     np.load('dev/cache_lt_y_A.npy')]
D = [p - core for p in P]
V_loc = np.array([[float(np.mean(D[i] * D[j])) for j in range(5)] for i in range(5)])
S0 = 1103.6568315036

ANCH = [
    (0.0300, 0.0000,  0.0000, 0.0000,  0.0000, 1104.8342852052),
    (0.1000, 0.0000,  0.0000, 0.0000,  0.0000, 1107.2877112561),
    (0.4800, 0.0000,  0.0000, 0.0000,  0.0000, 1113.4251423543),
    (0.4800, 0.1000,  0.0000, 0.0000,  0.0000, 1114.5296512406),
    (0.4944, 0.1030, -0.0300, 0.0000,  0.0000, 1115.0039993398),
    (0.5092, 0.1061, -0.0309, -0.0300, 0.0000, 1113.4528720829),
    (0.4671, 0.1817, -0.0316, 0.0000,  0.0000, 1115.1606262971),
    (0.4811, 0.1872, -0.0325, 0.0000, -0.0300, 1114.6410582665),
    (0.4381, 0.1740, -0.0316, 0.0354,  0.0350, 1115.4738393038),
]


def build_V(l1, l2):
    s = np.array([l1, l1, l1, l2, l2])
    return V_loc * np.sqrt(np.outer(s, s))


def fit(l1, l2):
    V = build_V(l1, l2)
    rows = [2 * np.array(a[:5]) for a in ANCH]
    rhs = [-(a[5] - S0) / K - float(np.array(a[:5]) @ V @ np.array(a[:5])) for a in ANCH]
    A, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    errs = [a[5] - (S0 - K * (2 * np.array(a[:5]) @ A + np.array(a[:5]) @ V @ np.array(a[:5])))
            for a in ANCH]
    return A, V, float(np.sum(np.square(errs))), errs


best = None
for l1 in np.linspace(0.3, 1.6, 66):
    for l2 in np.linspace(0.3, 6.0, 115):
        A_, V_, sse, errs = fit(l1, l2)
        if best is None or sse < best[0]:
            best = (sse, l1, l2, A_, V_, errs)
sse, l1, l2, A, V, errs = best
sc = lambda c: S0 - K * (2 * c @ A + c @ V @ c)
print(f'lam1={l1:.3f} lam2={l2:.3f} SSE={sse:.5f}')

# v128 기존5축 좌표의 예측치
c128 = np.array([0.4343, 0.1951, -0.0313, 0.0185, 0.0213])
S_base = sc(c128)
S_meas = 1115.6282950739
print(f'\nv128 기존5축 좌표 예측(S_base) = {S_base:.4f}')
print(f'v128 실측 = {S_meas:.4f}')
dev = S_meas - S_base
print(f'편차(= mc6aux+N1 번들효과 + 모델오차 ±0.12) = {dev:+.4f}')

# 번들 A 역산 (quad항은 미미하므로 선형근사, s_aux=0.0099, s_n1=0.0100)
# dev = -K(2*0.0099*A_aux + 2*0.01*A_n1 + 미미한 quad)
A_bundle = -dev / (K * 2 * 0.01)   # (A_aux+A_n1)/... 근사: 둘의 합
print(f'번들 실효 A(aux+N1 합 근사) = {A_bundle:+.3e}')
print(f'  (참고: mc6의 A = -5.04e-05. 번들 |A|가 그의 {abs(A_bundle)/5.04e-05*100:.0f}% 수준)')

# 남은 수확 계산: 기존5축 기준 최적점들
c124 = np.array([0.4671, 0.1817, -0.0316, 0.0, 0.0])
free = [0, 1, 3, 4]
fixed = np.zeros(5); fixed[2] = -0.0313
g = A + V @ fixed
c_optxu = fixed.copy(); c_optxu[free] = -np.linalg.solve(V[np.ix_(free, free)], g[free])
c_all = -np.linalg.solve(V, A)
print(f'\nxu고정 최적: ' + ' '.join(f'{NAMES[i]}={c_optxu[i]:+.4f}' for i in range(5))
      + f' -> 예측 {sc(c_optxu):.4f} (v128 대비 {sc(c_optxu)-S_meas:+.4f})')
print(f'전부자유 최적: ' + ' '.join(f'{NAMES[i]}={c_all[i]:+.4f}' for i in range(5))
      + f' -> 예측 {sc(c_all):.4f} (v128 대비 {sc(c_all)-S_meas:+.4f})')

print('\n앵커 재현오차:')
for a, e in zip(ANCH, errs):
    print(f'  {a[5]:.4f}  오차={e:+.4f}')
