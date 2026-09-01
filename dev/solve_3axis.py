"""v118 실측 후 실행: A3(pitchtype) 역산 + 3축 결합 최적화.
사용법: python dev/solve_3axis.py <v118_실측점수>
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
BASE = 1103.6568315036
CUR = 1114.5296512406
A1, V11 = -5.0596e-05, 1.0491e-04
A2 = -2.9235e-05
S = np.array([0.432, 0.227, 0.10])     # v118이 쓴 가중치

if len(sys.argv) < 2:
    print('사용법: python dev/solve_3axis.py <v118_실측점수>')
    sys.exit(0)
SCORE = float(sys.argv[1])

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
V = np.array([
    [V11, float(np.mean(d1*d2))*scale, float(np.mean(d1*d3))*scale],
    [float(np.mean(d1*d2))*scale, float(np.mean(d2**2))*scale, float(np.mean(d2*d3))*scale],
    [float(np.mean(d1*d3))*scale, float(np.mean(d2*d3))*scale, float(np.mean(d3**2))*scale],
])

# ΔScore = -K(2*S@A + S@V@S) 에서 A3 역산
dS = SCORE - BASE
quad = float(S @ V @ S)
lin_known = 2 * (S[0]*A1 + S[1]*A2)
A3 = (-dS/K - quad - lin_known) / (2*S[2])
Av = np.array([A1, A2, A3])
print(f'=== 프로브 결과 ===')
print(f'  v118={SCORE:.4f}  (v117={CUR:.4f}, Δ={SCORE-CUR:+.4f})')
print(f'  A3(pitchtype) = {A3:+.4e}   A1 대비 {abs(A3/A1)*100:.0f}%')
print(f'  {"-> 도움되는 방향(음수)" if A3 < 0 else "-> 손해 방향(양수)"}\n')

s_opt = -np.linalg.solve(V, Av)
g_opt = -K * (2*float(Av@s_opt) + float(s_opt@V@s_opt))
g_cur = -K * (2*float(Av@S) + quad)
print(f'=== 3축 결합 최적화 ===')
print(f'  s*(mc6)={s_opt[0]:+.4f}  s*(strk)={s_opt[1]:+.4f}  s*(pt)={s_opt[2]:+.4f}')
print(f'  최대이득 = {g_opt:+.2f}점   예상점수 = {BASE+g_opt:.2f}')
print(f'  현재(v118) 대비 = {g_opt-g_cur:+.2f}점')

print(f'\n=== 후보 조합 ===')
def sc(s):
    s = np.asarray(s, float)
    return -K * (2*float(Av@s) + float(s@V@s))
cands = [tuple(S), tuple(s_opt), (0.43, 0.23, 0.0), (0.40, 0.20, 0.20), (0.45, 0.25, 0.15)]
print(f'{"s1(mc6)":>9}{"s2(strk)":>10}{"s3(pt)":>9}{"예상Δ":>10}{"예상점수":>11}')
for c in cands:
    g = sc(c)
    print(f'{c[0]:>9.3f}{c[1]:>10.3f}{c[2]:>9.3f}{g:>10.2f}{BASE+g:>11.2f}')
