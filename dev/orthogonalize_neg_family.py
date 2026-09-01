"""음수축 가족에서 xgbunused(이미 v122에 반영됨)와 직교하는 성분만 추출해 남은 이득 계산.

핵심: d_perp = d_x - (V12/V11)*d_xu 에서 V12,V11은 y가 안 들어가는 순수 예측공간 양이라
      로컬 추정이 신뢰 가능(A와 달리). 직교화 후 rho가 남으면 그게 진짜 추가이득.
대조: 고정방향(사전학습 모델)의 rho는 n~25만에서 SE ~ 1/sqrt(n) = 0.002.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib, os

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

FAM = {
    'xgb_rawid':     'dev/cache_xgbrawid_{tag}.npy',
    'xgb_ctx':       'dev/cache_xgbctx_{tag}.npy',
    'xgb_hurdlectx': 'dev/cache_xgbhurdlectx_{tag}.npy',
    'lgbm_rawid':    'dev/cache_lgbmrawid_{tag}.npy',
    'cat_rawid':     'dev/cache_catrawid_{tag}.npy',
}


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
    n = len(yv)
    H = build8(tag)
    W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
    t0 = sum(W0.values()); W0 = {k: v / t0 for k, v in W0.items()}
    blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    p_xu = np.load(f'dev/cache_xgbunused_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    v117 = rest * blend8 + S_MC6 * p_mc6 + S_STRK * p_strk
    blend = np.clip(v117 * (1 - S_XU) + S_XU * p_xu, 0, 1)
    resid = yv - blend
    E_r2 = float(np.mean(resid ** 2))
    SE_rho = 1.0 / np.sqrt(n)

    d_xu = p_xu - blend; d_xu -= d_xu.mean()
    V11 = float(np.mean(d_xu ** 2))

    print(f'\n{"="*88}\n=== fold {tag} ({vs})   rho SE≈{SE_rho:.5f} (n={n:,}) ===\n{"="*88}')
    print(f'{"후보":<16}{"원래rho":>11}{"직교후rho":>12}{"직교후이득":>12}{"직교후s*":>11}{"z":>7}')
    for nm, tmpl in FAM.items():
        path = tmpl.format(tag=tag)
        if not os.path.exists(path):
            continue
        p = np.load(path)
        d = p - blend; d -= d.mean()
        V = float(np.mean(d ** 2))
        A = float(np.mean(d * (blend - yv)))
        rho0 = -A / np.sqrt(V * E_r2)

        # y-free 직교화
        V12 = float(np.mean(d * d_xu))
        d_perp = d - (V12 / V11) * d_xu
        d_perp -= d_perp.mean()
        Vp = float(np.mean(d_perp ** 2))
        if Vp < 1e-16:
            print(f'{nm:<16}{rho0:>+11.5f}   (완전중복)')
            continue
        Ap = float(np.mean(d_perp * (blend - yv)))
        rho_p = -Ap / np.sqrt(Vp * E_r2)
        gain_p = K * Ap ** 2 / Vp
        z = abs(rho_p) / SE_rho
        print(f'{nm:<16}{rho0:>+11.5f}{rho_p:>+12.5f}{gain_p:>+12.2f}{-Ap/Vp:>+11.4f}{z:>7.1f}')

    # 가족 전체를 xgbunused에 직교화한 뒤 균등번들
    ds = []
    for nm, tmpl in FAM.items():
        path = tmpl.format(tag=tag)
        if not os.path.exists(path):
            continue
        d = np.load(path) - blend; d -= d.mean()
        V12 = float(np.mean(d * d_xu))
        dp = d - (V12 / V11) * d_xu
        dp -= dp.mean()
        s = np.sqrt(np.mean(dp ** 2))
        if s > 1e-9:
            ds.append(dp / s)
    if ds:
        bundle = np.mean(ds, axis=0); bundle -= bundle.mean()
        Vb = float(np.mean(bundle ** 2))
        Ab = float(np.mean(bundle * (blend - yv)))
        rb = -Ab / np.sqrt(Vb * E_r2)
        print(f'\n  >> 직교화 균등번들: rho={rb:+.5f}  이득={K*Ab**2/Vb:+.2f}  '
              f's*={-Ab/Vb:+.4f}  z={abs(rb)/SE_rho:.1f}')
