"""v117 실측이 나오면 실행: strk의 A2를 역산하고 mc6와 결합 최적화.

v117 = v95*(1-s1-s2) + s1*mc6 + s2*strk,  s1=0.48, s2=0.10
v116 = v95*(1-s1)    + s1*mc6            (실측 1113.4251, Δ=+9.7683)

v117 - v116 의 차이는 s2*d2 를 더한 것:
  ΔBS = 2*s2*A2' + s2^2*V22,   A2' = A2 + s1*V12   (mc6가 이미 s1로 들어간 상태의 유효신호)
  => A2' = (-(Score117-Score116)/K - s2^2*V22) / (2*s2)

V22, V12는 로컬(fold A) 추정치를 mc6의 실측/로컬 V 비율로 스케일 보정해서 사용.
그다음 2x2 연립으로 (s1*, s2*) 결합 최적점을 구한다.

사용법: python dev/solve_joint_mc6_strk.py <v117_실측점수>
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
BASE = 1103.6568315036
SCORE116 = 1113.4251423543
S1, S2 = 0.48, 0.10
A1, V11 = -5.0596e-05, 1.0491e-04          # mc6 실측 확정값

if len(sys.argv) < 2:
    print('사용법: python dev/solve_joint_mc6_strk.py <v117_실측점수>')
    sys.exit(0)
SCORE117 = float(sys.argv[1])

# ---- 로컬에서 V22, V12 추정 후 mc6 스케일비로 보정 ----
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
va = season == 2024
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
V11_loc = float(np.mean(d1 ** 2))
V22_loc = float(np.mean(d2 ** 2))
V12_loc = float(np.mean(d1 * d2))
scale = V11 / V11_loc          # 실측/로컬 분산비
V22 = V22_loc * scale
V12 = V12_loc * scale
print(f'로컬: V11={V11_loc:.4e} V22={V22_loc:.4e} V12={V12_loc:.4e}  corr={V12_loc/np.sqrt(V11_loc*V22_loc):.3f}')
print(f'스케일보정({scale:.3f}배): V22={V22:.4e}  V12={V12:.4e}\n')

# ---- A2 역산 ----
dS = SCORE117 - SCORE116
A2p = (-dS / K - S2 ** 2 * V22) / (2 * S2)     # A2' = A2 + s1*V12
A2 = A2p - S1 * V12
print(f'=== 프로브 결과 ===')
print(f'  v116={SCORE116:.4f}  v117={SCORE117:.4f}  Δ={dS:+.4f}')
print(f'  A2\'(mc6 s=0.48 상태의 유효신호) = {A2p:+.4e}')
print(f'  A2(순수)                         = {A2:+.4e}')
print(f'  참고 A1 = {A1:+.4e}   -> strk가 mc6의 {abs(A2/A1)*100:.0f}% 크기\n')

if A2p >= 0:
    print('  A2\' >= 0 -> strk는 이 상태에서 도움 안 됨. s2=0 또는 음수가 최적.')
    s2_solo = -A2p / V22
    print(f'  (s1=0.48 고정 시 최적 s2 = {s2_solo:+.4f}, 이득 {K*A2p**2/V22:+.2f})')

# ---- 2x2 결합 최적화 ----
Vm = np.array([[V11, V12], [V12, V22]])
Av = np.array([A1, A2])
s_opt = -np.linalg.solve(Vm, Av)
gain = K * float(Av @ np.linalg.solve(Vm, Av))
print(f'=== 결합 최적화 ===')
print(f'  s1*(mc6) = {s_opt[0]:+.4f}   s2*(strk) = {s_opt[1]:+.4f}')
print(f'  최대이득 = {gain:+.2f}점   예상점수 = {BASE + gain:.2f}')
print(f'  (mc6 단독 최적 +9.77 대비 {gain-9.77:+.2f})')

print(f'\n=== 후보 가중치 조합별 예상 ===')
def sc(s1, s2):
    s = np.array([s1, s2])
    return -K * (2 * float(Av @ s) + float(s @ Vm @ s))
print(f'{"s1(mc6)":>9}{"s2(strk)":>10}{"예상Δ":>10}{"예상점수":>11}')
cands = [(S1, S2), (s_opt[0], s_opt[1]), (0.40, 0.30), (0.35, 0.40), (0.40, 0.40), (0.30, 0.45)]
for a, b in cands:
    g = sc(a, b)
    print(f'{a:>9.3f}{b:>10.3f}{g:>10.2f}{BASE+g:>11.2f}')
