"""오염 제거 최종 평가.
- 중심화(레벨 성분 제거) + 절편 b=0(레벨 복구 금지) -> 순수 방향성 기여만 측정
- fold A(2024)와 fold C(2022) 양쪽에서 fit->eval 정직 전이
- 대조군(랜덤/제로)이 0.0 근처로 나와야 절차가 건강한 것"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
df_raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['row_id', 'season'])


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


FILES = [
    ('phase3_preds', 'pred_tabm_base', 'pred_tabm_base'),
    ('phase3_preds', 'pred_tabm_pwl', 'pred_tabm_pwl'),
    ('phase3_preds', 'pred_embmlp_gated', 'pred'),
    ('phase3_preds', 'pred_embmlp_plain', 'pred'),
    ('phase4_preds', 'xgb_variants', 'pred_xgb_l2_ids'),
    ('phase4_preds', 'xgb_variants', 'pred_xgb_log_ids'),
    ('phase4_preds', 'xgb_variants', 'pred_xgb_l2_D'),
    ('phase4_preds', 'lgbm_variants', 'pred_A'),
]

rng = np.random.RandomState(0)
results = {}
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v88[f'{k}_weight']) for k in H}
    t = sum(W.values())
    W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)
    mth = X.loc[va, 'game_month'].to_numpy()
    H1 = mth <= 6
    H2 = ~H1
    rid = df_raw.loc[df_raw['season'] == vs, 'row_id'].to_numpy()
    pos = {r: i for i, r in enumerate(rid)}

    def honest(d):
        """fit구간에서 중심화 계수만 추정(절편 금지) -> eval구간 적용"""
        gains, coefs = [], []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            mdf = d[fit_m].mean()
            mrf = resid[fit_m].mean()
            cv = np.mean((d[fit_m] - mdf) * (resid[fit_m] - mrf))
            vr = np.mean((d[fit_m] - mdf) ** 2)
            a = cv / vr if vr > 1e-14 else 0.0
            bl = blend.copy()
            bl[ev_m] = blend[ev_m] + a * (d[ev_m] - mdf)
            gains.append(sc(bl, ev_m) - sc(blend, ev_m))
            coefs.append(a)
        return gains, coefs

    print(f'\n=== fold {tag} ({vs})  blend={sc(blend, np.ones(len(yv),bool)):.2f} ===')
    g, _ = honest(rng.normal(0, 0.05, len(yv)))
    print(f'  [대조군] 랜덤노이즈           H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')
    print(f'  {"후보":24s} {"단독":>9s} {"a(H1)":>8s} {"a(H2)":>8s} {"H1->H2":>8s} {"H2->H1":>8s} {"평균":>8s}')
    for d_dir, stem, col in FILES:
        f = f'dev/{d_dir}/fold_{vs}_{stem}.csv'
        if not os.path.exists(f):
            continue
        dd = pd.read_csv(f)
        if col not in dd.columns:
            continue
        idx = np.array([pos.get(r, -1) for r in dd['row_id'].to_numpy()])
        ok = idx >= 0
        p = np.full(len(blend), np.nan)
        p[idx[ok]] = dd[col].to_numpy()[ok]
        p = np.where(np.isnan(p), blend, p)
        gains, coefs = honest(p - blend)
        name = f'{stem}:{col}'[:24]
        results.setdefault(name, {})[tag] = np.mean(gains)
        print(f'  {name:24s} {sc(p, np.ones(len(yv),bool)):>9.1f} {coefs[0]:>+8.3f} {coefs[1]:>+8.3f} '
              f'{gains[0]:>+8.2f} {gains[1]:>+8.2f} {np.mean(gains):>+8.2f}')

print('\n=== 두 fold 모두에서 양수인 후보만 (진짜 후보) ===')
any_ok = False
for name, r in results.items():
    if len(r) == 2 and r['A'] > 0 and r['C'] > 0:
        print(f'  {name:26s} foldA={r["A"]:+7.2f}  foldC={r["C"]:+7.2f}')
        any_ok = True
if not any_ok:
    print('  없음')
