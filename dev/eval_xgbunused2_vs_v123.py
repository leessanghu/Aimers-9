"""xgbunused2를 v123(xgbunused+xgbrawid 이미 포함) 기준 직교화 평가 + 대조군 z검정. fold A만."""
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
S_MC6, S_STRK, S_XU, S_XR = 0.48, 0.10, -0.03, -0.03


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{m}.npy' for m in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{m}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{m}.npy') for m in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


tag, vs = 'A', 2024
va = season == vs
yv = y_all[va]
n = len(yv)
H = build8(tag)
W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
t0 = sum(W0.values()); W0 = {k: v / t0 for k, v in W0.items()}
blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
p_xu = np.load(f'dev/cache_xgbunused_{tag}.npy')
p_xr = np.load(f'dev/cache_xgbrawid_{tag}.npy')
rest = 1.0 - S_MC6 - S_STRK
v117 = rest * blend8 + S_MC6 * p_mc6 + S_STRK * p_strk
v122 = v117 * (1 - S_XU) + S_XU * p_xu
blend = np.clip(v122 * (1 - S_XR) + S_XR * p_xr, 0, 1)   # v123 근사(가중치 비례재정규화 생략, 근사충분)
resid = yv - blend
E_r2 = float(np.mean(resid ** 2))
SE_rho = 1.0 / np.sqrt(n)

p_xu2 = np.load('dev/cache_xgbunused2_A.npy')
d = p_xu2 - blend; d -= d.mean()
V = float(np.mean(d ** 2))
A = float(np.mean(d * (blend - yv)))
rho0 = -A / np.sqrt(V * E_r2)
gain0 = K * A ** 2 / V
print(f'원본 rho={rho0:+.5f}  이득={gain0:+.2f}  s*={-A/V:+.4f}  z(SE기준)={abs(rho0)/SE_rho:.1f}')

# xgbunused, xgbrawid에 이미 반영된 방향과 직교화
def orth(d, base_list):
    dp = d.copy()
    for b in base_list:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb < 1e-16:
            continue
        proj = float(np.mean(dp * b)) / Vb
        dp = dp - proj * b
    return dp - dp.mean()


d_xu = (p_xu - blend); d_xu -= d_xu.mean()
d_xr = (p_xr - blend); d_xr -= d_xr.mean()
d_perp = orth(d, [d_xu, d_xr])
Vp = float(np.mean(d_perp ** 2))
Ap = float(np.mean(d_perp * (blend - yv)))
rho_p = -Ap / np.sqrt(Vp * E_r2) if Vp > 1e-16 else 0.0
gain_p = K * Ap ** 2 / Vp if Vp > 1e-16 else 0.0
print(f'직교화후 rho={rho_p:+.5f}  이득={gain_p:+.2f}  s*={-Ap/Vp if Vp>1e-16 else 0:+.4f}  '
      f'z(SE기준)={abs(rho_p)/SE_rho:.1f}')

# 대조군: 무작위 순열로 같은 분산의 노이즈 벡터 20개, 같은 직교화+평가 절차
rng_rhos = []
for sd in range(20):
    rng = np.random.RandomState(2000 + sd)
    d_ctrl = rng.permutation(d)
    dp_c = orth(d_ctrl, [d_xu, d_xr])
    Vc = float(np.mean(dp_c ** 2))
    if Vc < 1e-16:
        continue
    Ac = float(np.mean(dp_c * (blend - yv)))
    rng_rhos.append(-Ac / np.sqrt(Vc * E_r2))
rng_rhos = np.array(rng_rhos)
z_ctrl = (abs(rho_p) - np.abs(rng_rhos).mean()) / (np.abs(rng_rhos).std(ddof=1) + 1e-18)
print(f'\n대조군(순열20) rho 평균={rng_rhos.mean():+.5f} SD={rng_rhos.std(ddof=1):.5f} '
      f'최대={np.abs(rng_rhos).max():.5f}')
print(f'실제 vs 대조군 z = {z_ctrl:.1f}  ->  {"진짜신호" if z_ctrl>3 else ("경계" if z_ctrl>1.5 else "허수(대조군수준)")}')
