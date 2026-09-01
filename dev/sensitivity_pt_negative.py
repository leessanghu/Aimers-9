"""s3=-0.14(pitchtype 음수) 추천의 근거를 흔들어본다.
A1,V11(mc6)은 실측 2점으로 정확히 확정. A2(strk)도 실측 1점으로 확정(주어진 V22,V12 하에).
A3(pt)도 실측 1점으로 확정(주어진 V13,V23,V33 하에).
문제: V13,V23,V33은 로컬(fold A) 추정치를 스칼라 하나로 스케일보정한 것 -> 검증 안 됨.
그 스케일이 축마다 다를 수 있다는 가정하에 V13/V23/V33을 개별적으로 흔들어
s3_opt와 예상점수가 얼마나 안정적인지 본다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
BASE = 1103.6568315036
A1, V11 = -5.0596e-05, 1.0491e-04
A2 = -2.9235e-05
A3 = -3.1903e-06

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
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
V22_0 = float(np.mean(d2 ** 2)) * scale
V33_0 = float(np.mean(d3 ** 2)) * scale
V12_0 = float(np.mean(d1 * d2)) * scale
V13_0 = float(np.mean(d1 * d3)) * scale
V23_0 = float(np.mean(d2 * d3)) * scale
Av = np.array([A1, A2, A3])

print(f'기준 스케일보정값: V22={V22_0:.3e} V33={V33_0:.3e} V12={V12_0:.3e} '
      f'V13={V13_0:.3e} V23={V23_0:.3e}\n')


def solve(V13, V23, V33):
    V = np.array([[V11, V12_0, V13], [V12_0, V22_0, V23], [V13, V23, V33]])
    try:
        s = -np.linalg.solve(V, Av)
    except np.linalg.LinAlgError:
        return None, None
    g = -K * (2 * float(Av @ s) + float(s @ V @ s))
    return s, g


print('=== V13(mc6-pt 공분산) 민감도 ± 50% ===')
for mult in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    s, g = solve(V13_0 * mult, V23_0, V33_0)
    print(f'  V13 x{mult:.2f}: s3*={s[2]:+.3f}  예상점수={BASE+g:.2f}')

print('\n=== V33(pt 자체분산) 민감도 ± 50% ===')
for mult in (0.5, 0.75, 1.0, 1.25, 1.5, 2.0):
    s, g = solve(V13_0, V23_0, V33_0 * mult)
    print(f'  V33 x{mult:.2f}: s3*={s[2]:+.3f}  예상점수={BASE+g:.2f}')

print('\n=== 종합 랜덤 스윕 (V13,V23,V33 각각 독립적으로 x0.5~x2.0) ===')
rng = np.random.RandomState(0)
res = []
for _ in range(2000):
    m13, m23, m33 = rng.uniform(0.5, 2.0, 3)
    s, g = solve(V13_0 * m13, V23_0 * m23, V33_0 * m33)
    if s is not None and np.all(np.abs(s) < 3):   # 발산 제거
        res.append((s[2], BASE + g))
res = np.array(res)
print(f'  유효 샘플 {len(res)}/2000')
print(f'  s3* 범위: {np.percentile(res[:,0],5):+.3f} ~ {np.percentile(res[:,0],95):+.3f}  '
      f'(중앙 {np.median(res[:,0]):+.3f})')
print(f'  예상점수 범위: {np.percentile(res[:,1],5):.2f} ~ {np.percentile(res[:,1],95):.2f}  '
      f'(중앙 {np.median(res[:,1]):.2f})')
neg_frac = (res[:, 0] < 0).mean()
print(f'  s3*가 음수로 나오는 비율: {neg_frac*100:.1f}%')

print('\n=== 비교: pt 그냥 빼기(s3=0) vs 실측 대비 ===')
s0, g0 = solve(0, 0, V33_0)   # V13=V23=0으로 만들면 s3 독립항만 남음 (참고용 아님, 그냥 표시)
g_drop = -K * (2*(A1*0.43+A2*0.23) + (0.43**2*V11+0.23**2*V22_0+2*0.43*0.23*V12_0))
print(f'  (0.43,0.23,0) 예상점수 = {BASE+g_drop:.2f}  <- 이미 실측 3점으로 검증된 안전영역')
