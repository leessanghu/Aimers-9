"""v122/v123/v125 세 프로브의 실측으로 각 축의 진짜 A를 역산하고, 부호역전의 공통원인 탐색.

가설: 실패한 두 프로브(xgb_rawid, lt_y)는 둘 다 '우리보다 약한 모델을 음수로 뺀' 것.
     실측이 반대라면 = '약한 모델을 더해야 한다' = 예측을 평균쪽으로 수축(shrink)해야 한다.
     즉 실제 test셋에서 우리 블렌드가 과신(over-confident)일 가능성.
검증: d_lty, d_xr 이 수축방향 -(blend - g)와 얼마나 정렬돼 있는지 측정.
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
W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
t_ = sum(W0.values()); W0 = {k: v / t_ for k, v in W0.items()}
core = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)

p_mc6 = np.load('dev/cache_mc6head_A.npy')
p_strk = np.load('dev/cache_strk_strk_linear_A.npy')
p_xu = np.load('dev/cache_xgbunused_A.npy')
p_xr = np.load('dev/cache_xgbrawid_A.npy')
p_lty = np.load('dev/cache_lt_y_A.npy')

# 정확한 가중치로 블렌드 재구성
v117 = 0.42 * core + 0.48 * p_mc6 + 0.10 * p_strk
v122 = 0.4326 * core + 0.4944 * p_mc6 + 0.1030 * p_strk - 0.0300 * p_xu
v124 = 0.3828 * core + 0.4671 * p_mc6 + 0.1817 * p_strk - 0.0316 * p_xu
for nm, b in [('v117', v117), ('v122', v122), ('v124', v124)]:
    s = 1e5 * (1 - np.mean((np.clip(b, 0, 1) - yv) ** 2) / B_)
    print(f'{nm} fold A BSS = {s:8.2f}   예측SD={b.std():.5f}  예측평균={b.mean():.5f}')

print(f'\n실제 성공률(fold A) = {yv.mean():.5f}')

# 실측 관측치로 A 역산: dScore = -K(2sA + s^2 V), V는 로컬(fold A)에서 직접 측정
OBS = [
    ('xgbunused  (v117->v122)', v117, p_xu, -0.03, 1115.0039993398 - 1114.5296512406),
    ('xgb_rawid  (v122->v123)', v122, p_xr, -0.03, 1113.4528720829 - 1115.0039993398),
    ('lt_y       (v124->v125)', v124, p_lty, -0.03, 1114.6410582665 - 1115.1606262971),
]
print(f'\n{"="*96}')
print(f'{"축":<26}{"V_local":>11}{"dScore":>10}{"A_real":>12}{"s*_real":>10}{"최대이득":>10}  판정')
print(f'{"="*96}')
results = {}
for nm, base, p, s_used, dS in OBS:
    d = p - base
    V = float(np.mean(d ** 2))          # 비중심(항등식과 일치)
    # dS = -K(2 s A + s^2 V)  ->  A = -(dS/K + s^2 V) / (2s)
    A = -(dS / K + s_used ** 2 * V) / (2 * s_used)
    s_star = -A / V
    gain = K * A ** 2 / V
    verdict = '빼는게 맞음' if s_star < 0 else '더하는게 맞음(우리가 반대로 감)'
    print(f'{nm:<26}{V:>11.3e}{dS:>+10.4f}{A:>+12.3e}{s_star:>+10.4f}{gain:>+10.2f}  {verdict}')
    results[nm] = (d, V, A, s_star, gain)

# 수축(shrinkage) 방향과의 정렬 검사
print(f'\n{"="*96}')
print('수축가설 검증: 각 축의 d가 수축방향 -(blend-g)와 얼마나 정렬돼 있나')
print(f'{"="*96}')
g = float(yv.mean())
for nm, base, p, s_used, dS in OBS:
    d = p - base
    shrink_dir = -(base - base.mean())     # 평균쪽으로 수축하는 방향
    c = float(np.corrcoef(d, shrink_dir)[0, 1])
    print(f'  {nm:<26} corr(d, 수축방향) = {c:+.4f}')

# 블렌드 자체의 로컬 캘리브레이션(참고): E[y|p] 기울기
print(f'\n{"="*96}')
print('참고: fold A에서 블렌드의 캘리브레이션 기울기 (1.0이면 완벽)')
print(f'{"="*96}')
for nm, b in [('v117', v117), ('v122', v122), ('v124', v124)]:
    bc = b - b.mean()
    slope = float(np.mean(bc * (yv - yv.mean())) / np.mean(bc ** 2))
    # 수축 최적계수 a* = slope, 그때 이득
    A_sh = float(np.mean((-(bc)) * (b - yv)))
    V_sh = float(np.mean(bc ** 2))
    print(f'  {nm}: 기울기={slope:.4f}   수축최적 a*={slope:.4f}  '
          f'로컬이득={K*A_sh**2/V_sh:+.2f}점  (기울기<1이면 과신=수축이 이득)')
