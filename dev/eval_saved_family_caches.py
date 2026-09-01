"""중단 전 저장된 mc6family/mc4f fold A 캐시를 v122 기준으로 평가(직교화+대조군).
학습 없음. 내일 프로덕션 후보 선별용."""
import sys, os
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
S_MC6, S_STRK, S_XU = 0.48, 0.10, -0.03

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
blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
p_mc6 = np.load('dev/cache_mc6head_A.npy')
p_strk = np.load('dev/cache_strk_strk_linear_A.npy')
p_xu = np.load('dev/cache_xgbunused_A.npy')
rest = 1.0 - S_MC6 - S_STRK
blend = np.clip((rest * blend8 + S_MC6 * p_mc6 + S_STRK * p_strk) * (1 - S_XU) + S_XU * p_xu, 0, 1)
resid = yv - blend
E_r2 = float(np.mean(resid ** 2))
SE = 1.0 / np.sqrt(len(yv))
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)


def orth(dd, bases):
    dp = dd.copy()
    for b in bases:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb < 1e-16:
            continue
        dp = dp - (float(np.mean(dp * b)) / Vb) * b
    return dp - dp.mean()


d_mc6 = p_mc6 - blend
d_xu = p_xu - blend

CANDS = {
    'A1_mc6aux':   'dev/mc6family_cache/A_mc6aux.npy',
    'A2_mc6brier': 'dev/mc6family_cache/A_mc6brier.npy',
    'A3_mc4':      'dev/mc6family_cache/A_mc4.npy',
    'mc4f':        'dev/mc4f_mc6pt_cache/A_mc4f.npy',
    'mc6pt':       'dev/mc4f_mc6pt_cache/A_mc6pt.npy',
}
for nm, path in CANDS.items():
    if not os.path.exists(path):
        print(f'{nm:<14} 캐시없음(미학습)')
        continue
    p = np.load(path)
    d = p - blend; d -= d.mean()
    V = float(np.mean(d ** 2)); A = float(np.mean(d * (blend - yv)))
    rho0 = -A / np.sqrt(V * E_r2)
    dp = orth(d, [d_mc6, d_xu])
    Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
    rho_p = -Ap / np.sqrt(Vp * E_r2) if Vp > 1e-16 else 0.0
    ctrl = []
    for sd in range(20):
        rng = np.random.RandomState(4000 + sd)
        dc = orth(rng.permutation(d), [d_mc6, d_xu])
        Vc = float(np.mean(dc ** 2))
        if Vc > 1e-16:
            ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
    ctrl = np.array(ctrl)
    z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
    verdict = '통과' if z > 3 else ('경계' if z > 1.5 else '허수')
    print(f'{nm:<14} BSS={sc(p):8.2f}  rho={rho0:+.5f}  직교후rho={rho_p:+.5f}  '
          f'이득={K*Ap**2/Vp if Vp>1e-16 else 0:+6.2f}  s*={-Ap/Vp if Vp>1e-16 else 0:+.4f}  z={z:5.1f}  {verdict}')
