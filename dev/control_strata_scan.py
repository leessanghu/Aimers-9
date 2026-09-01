"""층화축 스캔의 대조군 검증.

문제: 그룹평균 잔차를 H1에서 적합해 H2에 적용하는 건 '그룹별 절편 적합'이라
      [[h1h2-intercept-contamination]]이 경고한 허수이득 패턴. 신호가 0이어도
      그룹수만큼 자유도가 있어 rho가 양수로 나올 수 있다.

대조군: 완전 무작위 그룹(같은 그룹수)으로 동일 절차를 수행. 실제 층화축의 rho가
       대조군 분포를 유의하게 넘어야 진짜 신호.
시드 20개로 대조군 분포(평균/표준편차/최대)를 구하고 z-score로 판정.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')

HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
S_MC6, S_STRK, S_XU = 0.48, 0.10, -0.03
N_CTRL = 20


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


STRATA = [
    ('same_hand_x_gtype',   ['same_hand', 'cat_game_type'], 0),
    ('asof_batter_n_q10',   ['asof_batter_n'], 10),
    ('bat_inseason_q10',    ['bat_inseason_smooth'], 10),
    ('batter_id_count_q10', ['batter_id_count'], 10),
    ('x_ability_here_q10',  ['x_ability_here'], 10),
    ('inseason_succ_q10',   ['inseason_success_smooth'], 10),
]


def eval_grouping(grp, resid, blend, yv, H1, H2):
    """H1<->H2 양방향 그룹평균잔차 적용의 평균 rho / 이득."""
    rhos, gains = [], []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        gdf = pd.DataFrame({'g': grp[fit_m], 'r': resid[fit_m]})
        gm = gdf.groupby('g')['r'].mean()
        d_ev = pd.Series(grp[ev_m]).map(gm).fillna(0.0).to_numpy(np.float64)
        d_ev = d_ev - d_ev.mean()
        V = float(np.mean(d_ev ** 2))
        if V < 1e-16:
            continue
        A = float(np.mean(d_ev * (blend[ev_m] - yv[ev_m])))
        e2 = float(np.mean((yv[ev_m] - blend[ev_m]) ** 2))
        rhos.append(-A / np.sqrt(V * e2))
        gains.append(K * A ** 2 / V)
    if not rhos:
        return 0.0, 0.0
    return float(np.mean(rhos)), float(np.mean(gains))


for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
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

    mth = X.loc[va, 'game_month'].to_numpy()
    H1 = mth <= 6; H2 = ~H1
    n = len(yv)

    print(f'\n{"="*88}\n=== fold {tag} ({vs})  대조군 {N_CTRL}시드 ===\n{"="*88}')
    print(f'{"층화축":<24}{"실제rho":>11}{"대조평균":>11}{"대조SD":>10}{"대조최대":>10}{"z":>8}  판정')

    for nm, cols, nq in STRATA:
        if cols[0] == '__blend__':
            continue
        missing = [c for c in cols if c not in X.columns]
        if missing:
            continue
        if len(cols) == 1:
            key_src = X.loc[va, cols[0]].to_numpy(np.float64)
        else:
            arrs = [X.loc[va, c].to_numpy(np.float64) for c in cols]
            mults = np.cumprod([1] + [int(np.nanmax(a)) + 2 for a in arrs[:-1]])
            key_src = sum(a * m for a, m in zip(arrs, mults))
        if nq and nq > 0:
            ranks = pd.Series(key_src).rank(method='first', pct=True).to_numpy()
            grp = np.clip((ranks * nq).astype(int), 0, nq - 1)
        else:
            grp = pd.Series(key_src).fillna(-999).astype(np.int64).to_numpy()
        n_g = len(np.unique(grp))

        rho_real, gain_real = eval_grouping(grp, resid, blend, yv, H1, H2)

        # 대조군: 같은 그룹수, 같은 그룹크기분포를 갖는 무작위 배정
        ctrl_rhos = []
        for sd in range(N_CTRL):
            rng = np.random.RandomState(1000 + sd)
            perm = rng.permutation(n)
            grp_c = np.empty(n, dtype=np.int64)
            grp_c[perm] = grp          # 그룹크기 분포 보존, 배정만 무작위
            r_c, _ = eval_grouping(grp_c, resid, blend, yv, H1, H2)
            ctrl_rhos.append(r_c)
        ctrl_rhos = np.array(ctrl_rhos)
        mu, sd_ = ctrl_rhos.mean(), ctrl_rhos.std(ddof=1)
        z = (abs(rho_real) - abs(ctrl_rhos).mean()) / (abs(ctrl_rhos).std(ddof=1) + 1e-18)
        verdict = '진짜신호' if z > 3 else ('경계' if z > 1.5 else '대조군수준(허수)')
        print(f'{nm:<24}{rho_real:>+11.5f}{mu:>+11.5f}{sd_:>10.5f}'
              f'{np.abs(ctrl_rhos).max():>10.5f}{z:>8.1f}  {verdict}')
