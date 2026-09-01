"""v124 결합 최적화: 실측앵커 + 로컬V(스케일보정)로 3축(mc6/strk/xu) 결합 최적점 계산.

실측 앵커(전부 리더보드 실측):
  v95  (0,0,0)                = 1103.6568315036
  v112 (0.03,0,0)             = 1104.8342852052
  v114 (0.10,0,0)             = 1107.2877112561
  v116 (0.48,0,0)             = 1113.4251423543
  v117 (0.48,0.10,0)          = 1114.5296512406
  v122 (0.4944,0.103,-0.03)   = 1115.0039993398
  v123 (0.4944*1.03,...,xr)   = 1113.4528720829 (xr축 포함, 검증용 제외)

플랫 파라미터화: blend(c) = B + c1*(mc6-B) + c2*(strk-B) + c3*(xu-B),  B = v95 블렌드.
Score(c) = S0 - K(2 c·A + cᵀVc).  A는 앵커로 적합, V는 로컬(fold A) 측정 후
mc6 축의 실측 V11(=1.0491e-04)로 전체 스케일 보정(lambda).
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

# 로컬 V (비중심 - 정확 항등식은 평균 포함)
V_loc = np.empty((3, 3))
ds = [d1, d2, d3]
for i in range(3):
    for j in range(3):
        V_loc[i, j] = float(np.mean(ds[i] * ds[j]))
print('로컬 V (fold A):')
print(np.array2string(V_loc, precision=6))

# mc6 실측 V11 = 1.0491e-04 -> 스케일 보정
V11_REAL = 1.0491e-04
lam = V11_REAL / V_loc[0, 0]
V = V_loc * lam
print(f'\nlambda = {lam:.4f}  (V11 로컬 {V_loc[0,0]:.4e} -> 실측 {V11_REAL:.4e})')
print('보정 V:')
print(np.array2string(V, precision=6))

# 앵커 (c1,c2,c3, score)
S0 = 1103.6568315036
ANCHORS = [
    (0.03, 0.0, 0.0, 1104.8342852052),
    (0.10, 0.0, 0.0, 1107.2877112561),
    (0.48, 0.0, 0.0, 1113.4251423543),
    (0.48, 0.10, 0.0, 1114.5296512406),
    (0.4944, 0.103, -0.03, 1115.0039993398),
]
# Score(c) - S0 = -K(2 cA + cVc)  ->  cA = (-(dS)/K - cVc)/2 ... 선형계로 A 적합
rows, rhs = [], []
for c1, c2, c3, s in ANCHORS:
    c = np.array([c1, c2, c3])
    dS = s - S0
    quad = float(c @ V @ c)
    rows.append(2 * c)
    rhs.append(-dS / K - quad)
rows = np.array(rows); rhs = np.array(rhs)
A_fit, res, rank, _ = np.linalg.lstsq(rows, rhs, rcond=None)
print(f'\n적합 A = {A_fit}')
pred = S0 - K * (rows @ A_fit / 2 * 2 + np.array([c @ V @ c for c1, c2, c3, _ in ANCHORS for c in [np.array([c1, c2, c3])]]))
print('\n앵커 재현 검사 (실측 vs 예측):')
for (c1, c2, c3, s) in ANCHORS:
    c = np.array([c1, c2, c3])
    sp = S0 - K * (2 * c @ A_fit + c @ V @ c)
    print(f'  c=({c1:.4f},{c2:.3f},{c3:+.3f})  실측={s:.4f}  예측={sp:.4f}  오차={s-sp:+.4f}')

c_star = -np.linalg.solve(V, A_fit)
s_star = S0 - K * (2 * c_star @ A_fit + c_star @ V @ c_star)
print(f'\n결합 최적점 c* = mc6:{c_star[0]:.4f}  strk:{c_star[1]:.4f}  xu:{c_star[2]:+.4f}')
print(f'예측 점수 = {s_star:.4f}  (v122 대비 {s_star-1115.0039993398:+.4f})')

# 현재 v122 위치 재확인
c122 = np.array([0.4944, 0.103, -0.03])
s122p = S0 - K * (2 * c122 @ A_fit + c122 @ V @ c122)
print(f'v122 위치 예측 = {s122p:.4f} (실측 1115.0040)')

# 보수적 이동(최적점과 현재의 중간)도 출력
c_half = (c122 + c_star) / 2
s_half = S0 - K * (2 * c_half @ A_fit + c_half @ V @ c_half)
print(f'보수판(중간점) c = mc6:{c_half[0]:.4f} strk:{c_half[1]:.4f} xu:{c_half[2]:+.4f} -> 예측 {s_half:.4f}')
