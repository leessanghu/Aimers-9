"""9개 실측앵커로 5축(mc6/strk/xu/xr/lty) 결합 최적점을 정확히 푼다.

파라미터화: blend(c) = core + sum_i c_i (p_i - core),  core = v95 8헤드 정규화 블렌드.
Score(c) = S0 - K(2 c·A + cᵀVc).  V는 로컬(fold A) 비중심 측정 x 전역스케일 lam(그리드).
A는 9앵커 최소제곱. 앵커 재현오차로 lam 선택.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')

HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
va = season == 2024
yv = y_all[va]
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
m = len(NAMES)
V_loc = np.array([[float(np.mean(D[i] * D[j])) for j in range(m)] for i in range(m)])

S0 = 1103.6568315036
#        mc6     strk    xu       xr     lty     score
ANCH = [
    (0.0300, 0.0000,  0.0000, 0.0000,  0.0000, 1104.8342852052),
    (0.1000, 0.0000,  0.0000, 0.0000,  0.0000, 1107.2877112561),
    (0.4800, 0.0000,  0.0000, 0.0000,  0.0000, 1113.4251423543),
    (0.4800, 0.1000,  0.0000, 0.0000,  0.0000, 1114.5296512406),
    (0.4944, 0.1030, -0.0300, 0.0000,  0.0000, 1115.0039993398),
    (0.5092, 0.1061, -0.0309, -0.0300, 0.0000, 1113.4528720829),
    (0.4671, 0.1817, -0.0316, 0.0000,  0.0000, 1115.1606262971),
    (0.4811, 0.1872, -0.0325, 0.0000, -0.0300, 1114.6410582665),
]

best = None
for lam in np.linspace(0.2, 2.5, 116):
    V = V_loc * lam
    rows, rhs = [], []
    for *c, s in ANCH:
        c = np.array(c)
        rows.append(2 * c)
        rhs.append(-(s - S0) / K - float(c @ V @ c))
    A_fit, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    err = []
    for *c, s in ANCH:
        c = np.array(c)
        err.append(s - (S0 - K * (2 * c @ A_fit + c @ V @ c)))
    sse = float(np.sum(np.square(err)))
    if best is None or sse < best[0]:
        best = (sse, lam, A_fit, V, err)

sse, lam, A_fit, V, err = best
print(f'최적 전역스케일 lam = {lam:.3f}   SSE = {sse:.6f}   최대오차 = {max(abs(e) for e in err):.4f}')
print(f'\nA_real (축별):')
for i, nm in enumerate(NAMES):
    Vi = V[i, i]
    print(f'  {nm:<6} A={A_fit[i]:+.4e}  V={Vi:.4e}  단독s*={-A_fit[i]/Vi:+.4f}  단독최대이득={K*A_fit[i]**2/Vi:+.2f}')

print(f'\n앵커 재현:')
for (*c, s), e in zip(ANCH, err):
    c = np.array(c)
    print(f'  c=({c[0]:.4f},{c[1]:.4f},{c[2]:+.4f},{c[3]:+.4f},{c[4]:+.4f})  '
          f'실측={s:.4f}  오차={e:+.4f}')

c_star = -np.linalg.solve(V, A_fit)
s_star = S0 - K * (2 * c_star @ A_fit + c_star @ V @ c_star)
print(f'\n{"="*80}')
print(f'5축 결합 최적점:')
for i, nm in enumerate(NAMES):
    print(f'  {nm:<6} {c_star[i]:+.4f}')
print(f'  코어 = {1 - c_star.sum():+.4f}')
print(f'예측 점수 = {s_star:.4f}   (v124 실측 1115.1606 대비 {s_star - 1115.1606262971:+.4f})')

c124 = np.array([0.4671, 0.1817, -0.0316, 0.0, 0.0])
print(f'v124 위치 예측 = {S0 - K*(2*c124@A_fit + c124@V@c124):.4f}')

# 보수판: 최적점과 v124의 중간
for frac in (0.5, 0.7):
    ch = c124 + frac * (c_star - c124)
    sh = S0 - K * (2 * ch @ A_fit + ch @ V @ ch)
    print(f'\n보수판({int(frac*100)}%): ' + '  '.join(f'{NAMES[i]}={ch[i]:+.4f}' for i in range(m))
          + f'  코어={1-ch.sum():+.4f}  -> 예측 {sh:.4f}')
