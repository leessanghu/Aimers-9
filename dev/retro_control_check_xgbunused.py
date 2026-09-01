"""회고검증: xgbunused를 처음 스크리닝할 때(v117 기준, 직교화 없음) 순열대조군 검정을
했다면 통과했을지 확인. LGBM-mc6와 같은 기준으로 공정비교."""
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
S_MC6, S_STRK = 0.48, 0.10

for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    H = dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{q}.npy' for q in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{q}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{q}.npy') for q in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )
    W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
    t0 = sum(W0.values()); W0 = {k: v / t0 for k, v in W0.items()}
    blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    blend_v117 = np.clip(rest * blend8 + S_MC6 * p_mc6 + S_STRK * p_strk, 0, 1)
    resid = yv - blend_v117
    E_r2 = float(np.mean(resid ** 2))
    n = len(yv)
    SE = 1.0 / np.sqrt(n)

    p_xu = np.load(f'dev/cache_xgbunused_{tag}.npy')
    d = p_xu - blend_v117; d -= d.mean()
    V = float(np.mean(d ** 2)); A = float(np.mean(d * (blend_v117 - yv)))
    rho0 = -A / np.sqrt(V * E_r2)

    ctrl = []
    for sd in range(30):
        rng = np.random.RandomState(5000 + sd)
        dc = rng.permutation(d)
        Vc = float(np.mean(dc ** 2))
        Ac = float(np.mean(dc * (blend_v117 - yv)))
        ctrl.append(-Ac / np.sqrt(Vc * E_r2))
    ctrl = np.array(ctrl)
    z_ctrl = (abs(rho0) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
    z_se = abs(rho0) / SE
    print(f'fold {tag}: rho={rho0:+.5f}  z(SE기준)={z_se:.1f}  z(순열대조군)={z_ctrl:.1f}  '
          f'대조군SD={ctrl.std(ddof=1):.5f}  ->  {"통과" if z_ctrl>3 else ("경계" if z_ctrl>1.5 else "허수(대조군수준)")}')
