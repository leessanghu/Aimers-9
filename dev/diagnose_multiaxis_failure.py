"""다축 예측이 왜 빗나갔는지 진단 + 전체 실측으로 재적합.

핵심 문제: mc6 단독은 미지수 2개(A1,V11)를 실측 2점으로 정확히 풀었다 -> 예측 정확.
다축은 미지수 9개(A1,A2,A3,V11,V12,V13,V22,V23,V33)인데 실측은 6점뿐.
V12,V13,V22,V23,V33을 전부 '로컬 추정치 x 스칼라 0.703'으로 때웠고,
그 스칼라는 mc6 하나에서 나온 값이라 다른 축엔 안 맞을 수 있다.
=> v119 예측 1116.15 vs 실측 1113.14 (-3.0) 가 그 증거.

전체 실측 6점으로 자유파라미터를 재적합하고, 무엇을 알고 무엇을 모르는지 분리한다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from scipy.optimize import least_squares

B_ = 0.249807
K = 1e5 / B_
BASE = 1103.6568315036

# (s1_mc6, s2_strk, s3_pt, 실측점수)
OBS = [
    (0.030, 0.000,  0.000, 1104.8342852052),   # v112
    (0.100, 0.000,  0.000, 1107.2877112561),   # v114
    (0.480, 0.000,  0.000, 1113.4251423543),   # v116
    (0.480, 0.100,  0.000, 1114.5296512406),   # v117
    (0.432, 0.227,  0.100, 1112.8440148434),   # v118
    (0.478, 0.256, -0.140, 1113.1426492113),   # v119
]
LBL = ['v112', 'v114', 'v116', 'v117', 'v118', 'v119']

# mc6 단독 2점으로 A1,V11 정확히 확정 (이건 흔들리지 않는 사실)
s_a, d_a = OBS[0][0], OBS[0][3] - BASE
s_b, d_b = OBS[1][0], OBS[1][3] - BASE
M = np.array([[2*s_a, s_a**2], [2*s_b, s_b**2]])
A1, V11 = np.linalg.solve(M, np.array([-d_a/K, -d_b/K]))
print(f'=== 확정된 사실 (mc6 단독 2점, 흔들림 없음) ===')
print(f'  A1={A1:+.4e}  V11={V11:.4e}   s1*(단독)={-A1/V11:.4f}  최대이득={K*A1**2/V11:+.2f}\n')

# 로컬 추정치 (사전값)
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
sc0 = V11 / float(np.mean(d1**2))
prior = dict(
    V22=float(np.mean(d2**2))*sc0, V33=float(np.mean(d3**2))*sc0,
    V12=float(np.mean(d1*d2))*sc0, V13=float(np.mean(d1*d3))*sc0,
    V23=float(np.mean(d2*d3))*sc0,
)
print(f'=== 로컬 사전값 (스칼라 {sc0:.3f}배 보정) ===')
for k, v in prior.items():
    print(f'  {k}={v:+.4e}')

# 자유파라미터: A2,A3,V22,V33,V12,V13,V23 (7개) / 방정식 4개(v116~v119 중 v116은 중복이라 3개)
def predict(p, s):
    A2, A3, V22, V33, V12, V13, V23 = p
    A = np.array([A1, A2, A3])
    V = np.array([[V11, V12, V13], [V12, V22, V23], [V13, V23, V33]])
    s = np.asarray(s, float)
    return -K * (2*float(A@s) + float(s@V@s))


def resid(p, lam):
    out = []
    for (s1, s2, s3, sc_) in OBS:
        out.append(predict(p, (s1, s2, s3)) - (sc_ - BASE))
    # 사전값 정규화 (스케일 맞춰서)
    pri = np.array([prior['V22'], prior['V33'], prior['V12'], prior['V13'], prior['V23']])
    cur = np.array(p[2:])
    out.extend(list(lam * (cur - pri) / (np.abs(pri) + 1e-9)))
    return np.array(out)


p0 = np.array([-2.9e-05, -3.2e-06, prior['V22'], prior['V33'],
               prior['V12'], prior['V13'], prior['V23']])
for lam in (3.0, 1.0, 0.3):
    r = least_squares(resid, p0, args=(lam,), max_nfev=20000)
    A2, A3, V22, V33, V12, V13, V23 = r.x
    print(f'\n=== 재적합 (사전값 가중 lam={lam}) ===')
    print(f'  A2={A2:+.4e}  A3={A3:+.4e}')
    print(f'  V22={V22:+.4e}(사전 {prior["V22"]:+.3e})  V33={V33:+.4e}(사전 {prior["V33"]:+.3e})')
    print(f'  V12={V12:+.4e}(사전 {prior["V12"]:+.3e})  V13={V13:+.4e}(사전 {prior["V13"]:+.3e})'
          f'  V23={V23:+.4e}(사전 {prior["V23"]:+.3e})')
    print(f'  {"버전":<7}{"실측":>11}{"적합예측":>11}{"오차":>9}')
    for (s1, s2, s3, sc_), lb in zip(OBS, LBL):
        pr = BASE + predict(r.x, (s1, s2, s3))
        print(f'  {lb:<7}{sc_:>11.4f}{pr:>11.4f}{pr-sc_:>+9.4f}')
    A = np.array([A1, A2, A3])
    V = np.array([[V11, V12, V13], [V12, V22, V23], [V13, V23, V33]])
    ev = np.linalg.eigvalsh(V)
    if ev.min() > 0:
        s_opt = -np.linalg.solve(V, A)
        g = predict(r.x, s_opt)
        print(f'  최적 s*=({s_opt[0]:+.3f},{s_opt[1]:+.3f},{s_opt[2]:+.3f})  '
              f'예상={BASE+g:.2f}')
    else:
        print(f'  [경고] V가 양정부호 아님(고유값 최소 {ev.min():.2e}) -> 최적점 무의미')

print(f'\n=== 실측만으로 본 사실 ===')
for (s1, s2, s3, sc_), lb in zip(OBS, LBL):
    print(f'  {lb}: ({s1:.3f},{s2:.3f},{s3:+.3f}) -> {sc_:.4f}')
best = max(OBS, key=lambda o: o[3])
print(f'  >>> 최고점: {LBL[OBS.index(best)]} = {best[3]:.4f} '
      f'(s=({best[0]:.3f},{best[1]:.3f},{best[2]:+.3f}))')
