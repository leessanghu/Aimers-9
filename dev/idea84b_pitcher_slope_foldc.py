"""투수x2스트라이크 슬로프: K 더 키워서 최적점 찾기 + fold C(2022)로 이중검증.
모델 재학습 없는 순수 통계 correction이라 fold C도 저비용으로 만들 수 있다."""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'season', 'pitcher_id', 'strikes_before',
                          'control_success'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)
df['is_2k'] = (df['strikes_before'] == 2).astype(int)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
unc = 0.249807

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def build_v88raw(p):
    H = dict(
        base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{p}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{p}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )
    if p == 'A':
        P11 = np.load('dev/idea75_cache/A_proba11.npy')
        H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
        ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
        H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
        return sum(W[k] * H[k] for k in H)
    else:
        # fold C: mc5/ingame 캐시 없음 -> 8헤드로 재정규화
        keys8 = ['base','hurdle','multires','ordinal','midother','condball','countresid','future50']
        w8 = {k: W[k] for k in keys8}; t = sum(w8.values()); w8 = {k: v/t for k,v in w8.items()}
        return sum(w8[k] * H[k] for k in keys8)


def run_fold(p, train_upto, val_year):
    va = season == val_year
    yv = y[va]
    mth = X.loc[va, 'game_month'].to_numpy()
    v88_raw = build_v88raw(p)
    sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
    resid_base = yv - v88_raw

    train = df[df.season <= train_upto]
    g = train.groupby(['pitcher_id', 'is_2k'])['control_success'].agg(['sum', 'count']).unstack(fill_value=0)
    n2 = g[('count', 1)] if ('count', 1) in g.columns else pd.Series(0, index=g.index)
    s2 = g[('sum', 1)] if ('sum', 1) in g.columns else pd.Series(0, index=g.index)
    n0 = g[('count', 0)] if ('count', 0) in g.columns else pd.Series(0, index=g.index)
    s0 = g[('sum', 0)] if ('sum', 0) in g.columns else pd.Series(0, index=g.index)
    rate2 = (s2 / n2.replace(0, np.nan))
    rate0 = (s0 / n0.replace(0, np.nan))

    va_idx = df.index[va]
    pid_va = df.loc[va_idx, 'pitcher_id'].to_numpy()
    is2k_va = df.loc[va_idx, 'is_2k'].to_numpy()
    n2_va = n2.reindex(pid_va).fillna(0).to_numpy(np.float64)
    raw_gap = np.nan_to_num((rate2 - rate0).reindex(pid_va).to_numpy(np.float64), nan=0.0)

    H1 = mth <= 6; H2 = ~H1

    print(f'--- fold {p} (train<={train_upto} -> val {val_year}) ---')
    for K in [200, 400, 880, 1500, 2500, 4000, 6000, 10000]:
        shrunk = raw_gap * (n2_va / (n2_va + K))
        applied = np.where(is2k_va == 1, shrunk, 0.0)
        gains = []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            center = applied[fit_m].mean()
            cc = applied - center
            C = np.mean(cc[fit_m] * resid_base[fit_m])
            V = np.mean(cc[fit_m] ** 2)
            a = C / V if V > 1e-12 else 0.0
            adj = v88_raw.copy(); adj[ev_m] = v88_raw[ev_m] + a * cc[ev_m]
            gains.append(sc(adj, ev_m) - sc(v88_raw, ev_m))
        print(f'  K={K:5d}  std={applied.std():.5f}  H1->H2={gains[0]:+7.2f}  H2->H1={gains[1]:+7.2f}  평균={np.mean(gains):+7.2f}')
    print()


run_fold('A', 2023, 2024)
run_fold('C', 2021, 2022)
