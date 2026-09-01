"""안전판 비교: (a) 5축 전체 70% (b) xr/lty만 열기(mc6/strk/xu는 v124 고정).
V 추정오차 ±2배에 대한 민감도까지 같이 본다."""
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
m = 5
V_loc = np.array([[float(np.mean(D[i] * D[j])) for j in range(m)] for i in range(m)])
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
]
c124 = np.array([0.4671, 0.1817, -0.0316, 0.0, 0.0])


def fit(lam):
    V = V_loc * lam
    rows = [2 * np.array(a[:5]) for a in ANCH]
    rhs = [-(a[5] - S0) / K - float(np.array(a[:5]) @ V @ np.array(a[:5])) for a in ANCH]
    A, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    sse = sum((a[5] - (S0 - K * (2 * np.array(a[:5]) @ A + np.array(a[:5]) @ V @ np.array(a[:5])))) ** 2
              for a in ANCH)
    return A, V, sse


lam_best, A, V = None, None, None
for L in np.linspace(0.2, 2.5, 116):
    A_, V_, sse = fit(L)
    if lam_best is None or sse < lam_best[0]:
        lam_best, A, V = (sse, L), A_, V_
lam = lam_best[1]
score = lambda c: S0 - K * (2 * c @ A + c @ V @ c)
print(f'기준 lam={lam:.3f}   v124 예측={score(c124):.4f} (실측 1115.1606)')

# (a) 5축 전체 최적 x fraction
c_full = -np.linalg.solve(V, A)
print(f'\n[a] 5축 전체 최적 = ' + '  '.join(f'{NAMES[i]}={c_full[i]:+.4f}' for i in range(m)))
for f in (0.5, 0.7, 1.0):
    c = c124 + f * (c_full - c124)
    print(f'    {int(f*100):>3}% -> ' + '  '.join(f'{NAMES[i]}={c[i]:+.4f}' for i in range(m))
          + f'  코어={1-c.sum():+.4f}  예측={score(c):.4f}')

# (b) xr/lty만 최적화 (mc6/strk/xu는 v124 고정)
idx = [3, 4]
Vs = V[np.ix_(idx, idx)]
fixed = c124.copy()
grad = A + V @ fixed          # 고정축 영향 포함한 유효 A
c_sub = -np.linalg.solve(Vs, grad[idx])
c_b = fixed.copy(); c_b[idx] = c_sub
print(f'\n[b] xr/lty만 최적 (나머지 v124 고정)')
print(f'    최적 -> ' + '  '.join(f'{NAMES[i]}={c_b[i]:+.4f}' for i in range(m))
      + f'  코어={1-c_b.sum():+.4f}  예측={score(c_b):.4f}')
for f in (0.5, 0.7):
    c = c124.copy(); c[idx] = fixed[idx] + f * (c_sub - fixed[idx])
    print(f'    {int(f*100):>3}% -> ' + '  '.join(f'{NAMES[i]}={c[i]:+.4f}' for i in range(m))
          + f'  코어={1-c.sum():+.4f}  예측={score(c):.4f}')

# V 민감도: V가 실제로 2배/절반이면 위 후보들의 진짜 점수는?
print(f'\n{"="*84}\nV 민감도 (V가 추정보다 x0.5 / x2 인 경우 각 후보의 실제 점수)')
print(f'{"="*84}')
cands = {'v124(현행)': c124}
c = c124 + 0.7 * (c_full - c124); cands['[a]70%'] = c
c = c124.copy(); c[idx] = 0.7 * c_sub; cands['[b]70%'] = c
c = c124.copy(); c[idx] = c_sub; cands['[b]100%'] = c
for scale in (0.5, 1.0, 2.0):
    A2, V2, _ = fit(lam * scale)
    sc2 = lambda cc: S0 - K * (2 * cc @ A2 + cc @ V2 @ cc)
    print(f'  V x{scale:<4}: ' + '   '.join(f'{k}={sc2(v):.3f}' for k, v in cands.items()))
