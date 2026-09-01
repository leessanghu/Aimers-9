"""9앵커(v126 포함)로 5축 재적합. xr/lty가 음/양 양쪽 앵커를 갖게 되어 V가 처음으로 결정됨.
파라미터화: 전역 lam 대신 (코어축 lam1, 신규축 lam2) 2개 스케일도 시도."""
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
c126 = np.array([0.4381, 0.1740, -0.0316, 0.0354, 0.0350])
c124 = np.array([0.4671, 0.1817, -0.0316, 0.0, 0.0])


def build_V(l1, l2):
    s = np.array([l1, l1, l1, l2, l2])
    return V_loc * np.sqrt(np.outer(s, s))


def fit(V):
    rows = [2 * np.array(a[:5]) for a in ANCH]
    rhs = [-(a[5] - S0) / K - float(np.array(a[:5]) @ V @ np.array(a[:5])) for a in ANCH]
    A, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    errs = [a[5] - (S0 - K * (2 * np.array(a[:5]) @ A + np.array(a[:5]) @ V @ np.array(a[:5])))
            for a in ANCH]
    return A, float(np.sum(np.square(errs))), errs


best = None
for l1 in np.linspace(0.3, 1.6, 66):
    for l2 in np.linspace(0.3, 6.0, 115):
        V = build_V(l1, l2)
        A, sse, errs = fit(V)
        if best is None or sse < best[0]:
            best = (sse, l1, l2, A, V, errs)
sse, l1, l2, A, V, errs = best
print(f'최적 스케일: 코어축 lam1={l1:.3f}  신규축(xr,lty) lam2={l2:.3f}   SSE={sse:.5f}')
print(f'(v126 이전 추정은 lam=0.660 단일 -> 신규축 V를 {l2/0.660:.1f}배 과소평가했었음)\n')

print('A_real / 단독최적:')
for i, nm in enumerate(NAMES):
    print(f'  {nm:<5} A={A[i]:+.4e}  V={V[i,i]:.4e}  s*={-A[i]/V[i,i]:+.4f}  최대이득={K*A[i]**2/V[i,i]:+.2f}')

print('\n앵커 재현:')
for a, e in zip(ANCH, errs):
    print(f'  ({a[0]:.4f},{a[1]:.4f},{a[2]:+.4f},{a[3]:+.4f},{a[4]:+.4f}) 실측={a[5]:.4f} 오차={e:+.4f}')

sc = lambda c: S0 - K * (2 * c @ A + c @ V @ c)
print(f'\nv124 예측={sc(c124):.4f}(실측 1115.1606)   v126 예측={sc(c126):.4f}(실측 1115.4738)')

# xu 고정, 4축 최적
free = [0, 1, 3, 4]
fixed = np.zeros(5); fixed[2] = -0.0316
g = A + V @ fixed
c_opt = fixed.copy()
c_opt[free] = -np.linalg.solve(V[np.ix_(free, free)], g[free])
print(f'\n{"="*80}')
print('xu고정 4축 최적점: ' + '  '.join(f'{NAMES[i]}={c_opt[i]:+.4f}' for i in range(5))
      + f'  코어={1-c_opt.sum():+.4f}')
print(f'  예측 = {sc(c_opt):.4f}   (v126 대비 {sc(c_opt)-1115.4738393038:+.4f})')
for f in (0.5, 0.7, 1.0):
    c = c126.copy(); c[free] = c126[free] + f * (c_opt[free] - c126[free])
    print(f'  {int(f*100):>3}%: ' + '  '.join(f'{NAMES[i]}={c[i]:+.4f}' for i in range(5))
          + f'  예측={sc(c):.4f} ({sc(c)-1115.4738393038:+.4f})')

# 5축 전부 자유
c_all = -np.linalg.solve(V, A)
print(f'\n5축 전부자유 최적: ' + '  '.join(f'{NAMES[i]}={c_all[i]:+.4f}' for i in range(5))
      + f'  예측={sc(c_all):.4f} ({sc(c_all)-1115.4738393038:+.4f})')
