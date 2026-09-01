"""v122 기준선 위에서 (1) 음수가중치 가족의 상호독립성, (2) 새 층화축 잔차편향 스캔.

(1) xgbunused가 성공한 구조 = "거친 층화 평균을 과다반영하는 모델을 빼기".
    같은 구조의 후보(xgb_rawid/xgb_ctx/xgb_hurdlectx/lgbm_rawid/cat_rawid)가 서로
    독립이면 묶어서 한 번에 거둘 수 있다. d상관으로 확인.

(2) 학습 없이: train연도에서 그룹평균 잔차를 구해 검증연도에 적용 -> honest한
    "그룹별 레벨편향" 방향. fold A/C 부호일치 + rho 크기로 스크리닝.
    그룹은 전부 추론시점 가용 피처로만 정의(Rule4 안전).

이득공식: gain = K*rho^2*E[r^2] = 98,885 * rho^2  (rho=잔차상관)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib, os

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


NEG_FAMILY = {
    'xgbunused':     'dev/cache_xgbunused_{tag}.npy',
    'xgb_rawid':     'dev/cache_xgbrawid_{tag}.npy',
    'xgb_ctx':       'dev/cache_xgbctx_{tag}.npy',
    'xgb_hurdlectx': 'dev/cache_xgbhurdlectx_{tag}.npy',
    'lgbm_rawid':    'dev/cache_lgbmrawid_{tag}.npy',
    'cat_rawid':     'dev/cache_catrawid_{tag}.npy',
}

# 층화축 후보: (이름, 컬럼리스트, 연속형이면 분위수 개수)
STRATA = [
    ('asof_pitcher_n_q10',     ['asof_pitcher_n'], 10),
    ('asof_batter_n_q10',      ['asof_batter_n'], 10),
    ('pitcher_id_count_q10',   ['pitcher_id_count'], 10),
    ('batter_id_count_q10',    ['batter_id_count'], 10),
    ('inseason_succ_q10',      ['inseason_success_smooth'], 10),
    ('bat_inseason_q10',       ['bat_inseason_smooth'], 10),
    ('x_ability_here_q10',     ['x_ability_here'], 10),
    ('inseason_cmd_q10',       ['inseason_cmd_index'], 10),
    ('game_month',             ['game_month'], 0),
    ('count_state',            ['count_state'], 0),
    ('balls_x_strikes',        ['balls_before', 'strikes_before'], 0),
    ('same_hand_x_gtype',      ['same_hand', 'cat_game_type'], 0),
    ('season_x_month',         ['season', 'game_month'], 0),
    ('pitcher_team',           ['pitcher_team_id_count'], 10),
    ('BLEND_DECILE(캘리브)',    ['__blend__'], 20),
]

results_neg = {}
results_str = {}

for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    tr = season <= upto
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
    # v122 = v117 전부 x1.03 + xgbunused x(-0.03)
    blend = np.clip(v117 * (1 - S_XU) + S_XU * p_xu, 0, 1)
    resid = yv - blend           # r = y - blend
    E_r2 = float(np.mean(resid ** 2))

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
    print(f'\n{"="*92}\n=== fold {tag} ({vs})   v122기준선 BSS={sc(blend):.2f} ===\n{"="*92}')

    # ---------- (1) 음수축 가족 ----------
    D = {}
    print('  [음수축 가족] rho / 최대이득 / 최적가중치')
    for nm, tmpl in NEG_FAMILY.items():
        path = tmpl.format(tag=tag)
        if not os.path.exists(path):
            continue
        p = np.load(path)
        d = p - blend; d = d - d.mean()
        V = float(np.mean(d ** 2))
        if V < 1e-14:
            continue
        A = float(np.mean(d * (blend - yv)))
        rho = -A / np.sqrt(V * E_r2)
        D[nm] = d
        results_neg.setdefault(nm, {})[tag] = (rho, K * A ** 2 / V, -A / V)
        print(f'    {nm:<16} rho={rho:+.5f}  최대이득={K*A**2/V:+7.2f}  s*={-A/V:+.4f}')

    if len(D) > 1:
        names = list(D)
        M = np.column_stack([D[n] for n in names])
        C = np.corrcoef(M.T)
        print(f'\n  [음수축 상호 d상관] (1에 가까우면 중복, 낮으면 묶어서 이득 가산)')
        print('    ' + ' ' * 16 + ''.join(f'{n[:8]:>10}' for n in names))
        for i, n in enumerate(names):
            print(f'    {n:<16}' + ''.join(f'{C[i,j]:>10.3f}' for j in range(len(names))))

        # 균등가중 번들(피팅 없음)의 rho
        bundle = np.mean([D[n] / np.sqrt(np.mean(D[n]**2)) for n in names], axis=0)
        bundle -= bundle.mean()
        Vb = float(np.mean(bundle ** 2))
        Ab = float(np.mean(bundle * (blend - yv)))
        rb = -Ab / np.sqrt(Vb * E_r2)
        print(f'\n    >> 균등번들(정규화 후 평균): rho={rb:+.5f}  최대이득={K*Ab**2/Vb:+7.2f}  s*={-Ab/Vb:+.4f}')
        results_neg.setdefault('__BUNDLE__', {})[tag] = (rb, K * Ab ** 2 / Vb, -Ab / Vb)

    # ---------- (2) 층화축 잔차편향 (학습 없음, honest) ----------
    print(f'\n  [층화축 잔차편향] train연도 그룹평균잔차 -> 검증연도 적용')
    resid_tr_full = np.full(len(y_all), np.nan)
    # train 잔차는 fold별 캐시가 없으므로, 그룹평균은 '검증연도 외' 데이터로 못 구함.
    # 대신 honest 대용: 검증연도를 전반기/후반기로 나눠 H1에서 그룹평균 -> H2 적용, 양방향.
    mth = X.loc[va, 'game_month'].to_numpy()
    H1 = mth <= 6; H2 = ~H1

    for nm, cols, nq in STRATA:
        if cols == ['__blend__']:
            key_src = blend
        else:
            missing = [c for c in cols if c not in X.columns]
            if missing:
                print(f'    {nm:<24} [skip] 컬럼없음 {missing}')
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

        # 양방향 honest: H1에서 그룹평균잔차 -> H2에 적용, 그리고 반대
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
            continue
        rm, gm_ = float(np.mean(rhos)), float(np.mean(gains))
        results_str.setdefault(nm, {})[tag] = (rm, gm_)
        print(f'    {nm:<24} rho={rm:+.5f}  n_grp={len(np.unique(grp)):>4}  이득추정={gm_:+7.2f}')

print(f'\n{"="*92}\n=== 종합: fold A/C 부호일치 검사 ===\n{"="*92}')
print('[음수축 가족]')
for nm, d in results_neg.items():
    if 'A' in d and 'C' in d:
        rA, gA, sA = d['A']; rC, gC, sC = d['C']
        ok = '동일부호' if rA * rC > 0 else '부호반전'
        print(f'  {nm:<16} A:rho={rA:+.5f} s*={sA:+.4f} | C:rho={rC:+.5f} s*={sC:+.4f}  -> {ok}')
print('\n[층화축]')
rows = []
for nm, d in results_str.items():
    if 'A' in d and 'C' in d:
        rA, gA = d['A']; rC, gC = d['C']
        ok = rA * rC > 0
        rows.append((nm, rA, rC, gA, gC, ok))
rows.sort(key=lambda t: -min(abs(t[1]), abs(t[2])) if t[5] else 0)
for nm, rA, rC, gA, gC, ok in rows:
    print(f'  {nm:<24} A:rho={rA:+.5f}({gA:+6.2f}) | C:rho={rC:+.5f}({gC:+6.2f})  -> '
          f'{"동일부호" if ok else "부호반전"}')
