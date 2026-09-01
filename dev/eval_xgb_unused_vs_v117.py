import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
NEED_RHO = 0.01740
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')

HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
S_MC6, S_STRK = 0.48, 0.10


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


for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    H = build8(tag)
    W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
    t0 = sum(W0.values()); W0 = {k: v / t0 for k, v in W0.items()}
    blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    blend_v117 = np.clip(rest * blend8 + S_MC6 * p_mc6 + S_STRK * p_strk, 0, 1)
    resid = yv - blend_v117

    p = np.load(f'dev/cache_xgbunused_{tag}.npy')
    d = p - blend_v117; dc = d - d.mean()
    V = float(np.mean(dc ** 2)); A = float(np.mean(dc * (blend_v117 - yv)))
    rho = -A / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
    gain = K * A ** 2 / V if V > 1e-14 else 0.0
    s_star = -A / V if V > 1e-14 else 0.0
    print(f'fold {tag}: rho={rho:+.5f} ({abs(rho)/NEED_RHO*100:5.1f}%)  로컬최대이득={gain:+.2f}  '
          f'A={A:+.3e}  V={V:.3e}  s*={s_star:+.4f}')
