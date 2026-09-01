"""신규 handoff 피처(li_resid, team_matchup) 정식검증.
3종 세트: 대조군 / 중심화+무절편 / fold A+C 재현."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

feat_A = pd.read_parquet('dev/handoff_features_foldA.parquet')


def build_li_resid(train_mask):
    df = raw_all
    tr = df[train_mask].copy()
    tr['inning_c'] = tr['inning'].clip(upper=10)
    tr['sd_c'] = tr['score_diff_home'].clip(-6, 6)
    key_cols = ['inning_c', 'outs_before', 'base_state', 'sd_c']
    tab = tr.groupby(key_cols)['li'].agg(['mean', 'count']).reset_index()
    global_li = float(tr['li'].mean())
    K = 30.0
    tab['li_expect'] = (tab['count'] * tab['mean'] + K * global_li) / (tab['count'] + K)
    full = df.copy()
    full['inning_c'] = full['inning'].clip(upper=10)
    full['sd_c'] = full['score_diff_home'].clip(-6, 6)
    merged = full.merge(tab[key_cols + ['li_expect']], on=key_cols, how='left')
    merged['li_expect'] = merged['li_expect'].fillna(global_li)
    return (full['li'].to_numpy(np.float64) - merged['li_expect'].to_numpy(np.float64))


def build_team_matchup(train_mask):
    df = raw_all
    tr = df[train_mask]
    tab = tr.groupby(['pitcher_team_id', 'batter_team_id'])['control_success'].agg(['mean', 'count']).reset_index()
    global_y = float(tr['control_success'].mean())
    K = 500.0
    tab['te'] = (tab['count'] * tab['mean'] + K * global_y) / (tab['count'] + K)
    merged = df.merge(tab[['pitcher_team_id', 'batter_team_id', 'te']],
                       on=['pitcher_team_id', 'batter_team_id'], how='left')
    merged['te'] = merged['te'].fillna(global_y)
    return merged['te'].to_numpy(np.float64) - global_y


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


for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v88[f'{k}_weight']) for k in H}
    t = sum(W.values())
    W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    raw = raw_all[raw_all['season'] == vs].reset_index(drop=True)

    li_resid = build_li_resid(season <= upto)[season == vs]
    matchup = build_team_matchup(season <= upto)[season == vs]

    resid = yv - blend
    mth = raw['game_month'].to_numpy()
    H1 = mth <= 6
    H2 = ~H1
    sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / 0.249807)
    rng = np.random.RandomState(5)

    def run(dd):
        gains, coefs = [], []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            mdf = dd[fit_m].mean(); mrf = resid[fit_m].mean()
            cv = np.mean((dd[fit_m] - mdf) * (resid[fit_m] - mrf))
            vr = np.mean((dd[fit_m] - mdf) ** 2)
            a = cv / vr if vr > 1e-14 else 0.0
            bl = blend.copy()
            bl[ev_m] = blend[ev_m] + a * (dd[ev_m] - mdf)
            gains.append(sc(bl, ev_m) - sc(blend, ev_m))
            coefs.append(a)
        return gains, coefs

    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    ctrl1 = rng.normal(0, li_resid.std(), len(yv))
    gc1, _ = run(ctrl1)
    print(f'  [li_resid] 대조군 평균={np.mean(gc1):+7.2f}')
    g1, c1 = run(li_resid)
    print(f'  [li_resid] 신호   H1->H2={g1[0]:+7.2f} H2->H1={g1[1]:+7.2f} 평균={np.mean(g1):+7.2f}  a={c1[0]:+.4f}/{c1[1]:+.4f}')

    ctrl2 = rng.normal(0, matchup.std(), len(yv))
    gc2, _ = run(ctrl2)
    print(f'  [team_matchup] 대조군 평균={np.mean(gc2):+7.2f}')
    g2, c2 = run(matchup)
    print(f'  [team_matchup] 신호   H1->H2={g2[0]:+7.2f} H2->H1={g2[1]:+7.2f} 평균={np.mean(g2):+7.2f}  a={c2[0]:+.4f}/{c2[1]:+.4f}')
