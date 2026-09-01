"""v124 실측 반영 후 3축(mc6/strk/xu) 재적합. 6개 앵커로 A 재적합(V는 우선 고정).
잔차 패턴을 보고 V(특히 strk 관련 성분)가 잘못됐는지 진단."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')

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
W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
t_ = sum(W0.values()); W0 = {k: v / t_ for k, v in W0.items()}
B = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)

d1 = np.load('dev/cache_mc6head_A.npy') - B
d2 = np.load('dev/cache_strk_strk_linear_A.npy') - B
d3 = np.load('dev/cache_xgbunused_A.npy') - B
ds = [d1, d2, d3]
V_loc = np.array([[float(np.mean(ds[i]*ds[j])) for j in range(3)] for i in range(3)])
V11_REAL = 1.0491e-04
lam = V11_REAL / V_loc[0, 0]
V_old = V_loc * lam

S0 = 1103.6568315036
ANCHORS6 = [
    (0.03, 0.0, 0.0, 1104.8342852052),
    (0.10, 0.0, 0.0, 1107.2877112561),
    (0.48, 0.0, 0.0, 1113.4251423543),
    (0.48, 0.10, 0.0, 1114.5296512406),
    (0.4944, 0.103, -0.03, 1115.0039993398),
    (0.4671, 0.1817, -0.0316, 1115.1606262971),
]


def fit_and_report(V, anchors, label):
    rows, rhs = [], []
    for c1, c2, c3, s in anchors:
        c = np.array([c1, c2, c3])
        dS = s - S0
        rows.append(2 * c)
        rhs.append(-dS / K - float(c @ V @ c))
    rows = np.array(rows); rhs = np.array(rhs)
    A_fit, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    print(f'\n=== {label} ===')
    print(f'A_fit = {A_fit}')
    max_err = 0.0
    for (c1, c2, c3, s) in anchors:
        c = np.array([c1, c2, c3])
        sp = S0 - K * (2 * c @ A_fit + c @ V @ c)
        err = s - sp
        max_err = max(max_err, abs(err))
        print(f'  c=({c1:.4f},{c2:.4f},{c3:+.4f})  실측={s:.4f}  예측={sp:.4f}  오차={err:+.4f}')
    print(f'  최대오차 = {max_err:.4f}')
    if np.linalg.det(V) != 0:
        c_star = -np.linalg.solve(V, A_fit)
        s_star = S0 - K * (2 * c_star @ A_fit + c_star @ V @ c_star)
        print(f'  결합최적점 c* = mc6:{c_star[0]:+.4f} strk:{c_star[1]:+.4f} xu:{c_star[2]:+.4f}  예측={s_star:.4f}')
    return A_fit


fit_and_report(V_old, ANCHORS6, '기존 V(고정)로 6앵커 재적합')

# strk의 V22, V12, V23를 자유 파라미터로 - v124가 6번째 점을 주니 strk 관련 정보 갱신
# 간단화: strk축만 스케일 rho 도입 (V_strk_row *= mu), mc6/xu 서로간은 고정
best = None
for mu in np.linspace(0.1, 3.0, 30):
    Vm = V_old.copy()
    Vm[1, :] *= mu; Vm[:, 1] *= mu
    Vm[1, 1] = V_old[1, 1] * mu * mu
    rows, rhs = [], []
    for c1, c2, c3, s in ANCHORS6:
        c = np.array([c1, c2, c3])
        dS = s - S0
        rows.append(2 * c)
        rhs.append(-dS / K - float(c @ Vm @ c))
    rows = np.array(rows); rhs = np.array(rhs)
    A_fit, res, *_ = np.linalg.lstsq(rows, rhs, rcond=None)
    pred = np.array([S0 - K * (2 * np.array([c1, c2, c3]) @ A_fit + np.array([c1, c2, c3]) @ Vm @ np.array([c1, c2, c3]))
                      for c1, c2, c3, s in ANCHORS6])
    real = np.array([s for *_, s in ANCHORS6])
    sse = float(np.sum((pred - real) ** 2))
    if best is None or sse < best[0]:
        best = (sse, mu, A_fit, Vm)

sse, mu, A_fit, Vm = best
print(f'\n=== strk V 스케일(mu={mu:.3f}) 그리드서치 최적 (SSE={sse:.6f}) ===')
fit_and_report(Vm, ANCHORS6, f'mu={mu:.3f} 최적')
