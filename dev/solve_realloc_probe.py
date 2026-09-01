"""v121(t=0.12) 실측으로 재배분 축의 A,V 역산 + 최적 t 추정."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
V117 = 1114.5296512406
V121 = 1114.441248282
T = 0.12
DELTA = V121 - V117
print(f'v117={V117:.4f}  v121={V121:.4f}  Δ={DELTA:+.4f}\n')

# 로컬 V 추정 (fold A, t=0.12에서의 d = blend_t - blend0 분산을 mc6 실측스케일로 보정)
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
OVERLAP = ['midother', 'condball', 'countresid', 'future50']
INDEP = ['base', 'hurdle', 'ordinal']
S_MC6, S_STRK = 0.48, 0.10


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


def make_blend(H, W8, tag):
    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    base8 = sum(W8[k] * H[k] for k in HEADS8)
    return np.clip(rest * base8 + S_MC6 * p_mc6 + S_STRK * p_strk, 0, 1)


def weights_at_t(W0, t):
    Wt = dict(W0)
    overlap_sum = sum(W0[k] for k in OVERLAP)
    indep_sum = sum(W0[k] for k in INDEP)
    move = overlap_sum * t
    for k in OVERLAP:
        Wt[k] = W0[k] * (1 - t)
    for k in INDEP:
        Wt[k] = W0[k] + move * (W0[k] / indep_sum)
    return Wt


W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
t0 = sum(W0.values()); W0 = {k: v / t0 for k, v in W0.items()}

# mc6 실측 스케일 보정계수 (오늘 확립한 절차)
A1_real, V11_real = -5.0596e-05, 1.0491e-04
va_mc6 = season == 2024
H_mc6 = build8('A')
Wmc6 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
t_ = sum(Wmc6.values()); Wmc6 = {k: v / t_ for k, v in Wmc6.items()}
blend_mc6 = np.clip(sum(Wmc6[k] * H_mc6[k] for k in HEADS8), 0, 1)
d_mc6_local = np.load('dev/cache_mc6head_A.npy') - blend_mc6
scale = V11_real / float(np.mean(d_mc6_local ** 2))
print(f'실측스케일 보정계수 = {scale:.4f}\n')

Vs = []
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    H = build8(tag)
    blend0 = make_blend(H, W0, tag)
    blend_t = make_blend(H, weights_at_t(W0, T), tag)
    d = blend_t - blend0
    d = d - d.mean()
    V_loc = float(np.mean(d ** 2)) * scale
    Vs.append(V_loc)
    print(f'fold{tag}: V(로컬,스케일보정) = {V_loc:.4e}')

V_use = float(np.mean(Vs))
print(f'\n평균 V = {V_use:.4e}')

# A 역산: DeltaScore = -K*(2*T*A + T^2*V)
A_real = (-DELTA / K - T ** 2 * V_use) / (2 * T)
s_star = -A_real / V_use
gain_max = K * A_real ** 2 / V_use
print(f'\n=== 역산 결과 ===')
print(f'  A(재배분) = {A_real:+.4e}')
print(f'  s*(최적 t) = {s_star:+.4f}')
print(f'  최대이득 = {gain_max:+.2f}점')
print(f'\n  참고: mc6 A1={A1_real:+.4e}, strk A2=-2.9235e-05')
print(f'  재배분 A는 그 대비 {abs(A_real/A1_real)*100:.1f}%(mc6) / {abs(A_real/-2.9235e-05)*100:.1f}%(strk)')

print(f'\n=== t별 예상 ΔScore ===')
for t in (0.0, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.30):
    g = -K * (2 * t * A_real + t ** 2 * V_use)
    print(f'  t={t:.2f}: 예상Δ={g:+.2f}  예상점수={V117+g:.2f}')
