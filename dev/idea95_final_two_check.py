import sys
sys.path.insert(0, 'dev/teammate_v1')
import numpy as np, pandas as pd, joblib
import pipeline as P
df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
unc = 0.249807
tr = season <= 2023
train_df = df.loc[tr].reset_index(drop=True)
tm = P._load_tm()
tables = P.build_tables(train_df)
df_fe = P.apply_fe(df, tm, tables)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
p = 'A'
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
P11 = np.load('dev/idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
raw = sum(W[k] * H[k] for k in H)
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
v88_final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)
sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
resid = yv - v88_final
H1 = mth <= 6
H2 = ~H1

v1 = np.nan_to_num(df_fe.loc[va, 'pitcher_middle_momentum'].to_numpy(np.float64), nan=0.0)
v2 = np.nan_to_num(df_fe.loc[va, 'asof_pitcher_success_shrunk'].to_numpy(np.float64), nan=0.0)

print('corr(둘 사이) =', np.corrcoef(v1, v2)[0, 1])
Xv = X.loc[va]
for c in ['asof_pitcher_success_rate_smooth', 'inseason_success_smooth', 'x_ability_here']:
    if c in Xv.columns:
        print(f'corr(shrunk, {c}) = {np.corrcoef(v2, Xv[c].to_numpy(np.float64))[0,1]:.4f}')
print()


def bucket_corr(v, fit_m, ev_m, nbin=8):
    edges = np.unique(np.quantile(v[fit_m], np.linspace(0, 1, nbin + 1)))
    edges = edges.astype(float); edges[0] -= 1e-9; edges[-1] += 1e-9
    bf_ = np.clip(np.digitize(v[fit_m], edges) - 1, 0, len(edges) - 2)
    be = np.clip(np.digitize(v[ev_m], edges) - 1, 0, len(edges) - 2)
    return bf_, be, len(edges) - 1


print('=== 순차적용 (momentum 먼저, 그 다음 shrunk를 잔차에) ===')
gains = []
for fit_m, ev_m in [(H1, H2), (H2, H1)]:
    bf1, be1, nb1 = bucket_corr(v1, fit_m, ev_m)
    rf = resid[fit_m]; gl = rf.mean()
    cmap1 = np.zeros(nb1)
    for b in range(nb1):
        m = bf1 == b
        if m.sum() >= 500:
            cmap1[b] = rf[m].mean() - gl
    adj = v88_final.copy(); adj[ev_m] = v88_final[ev_m] + cmap1[be1]
    resid2 = yv - adj
    bf2, be2, nb2 = bucket_corr(v2, fit_m, ev_m)
    rf2 = resid2[fit_m]; gl2 = rf2.mean()
    cmap2 = np.zeros(nb2)
    for b in range(nb2):
        m = bf2 == b
        if m.sum() >= 500:
            cmap2[b] = rf2[m].mean() - gl2
    adj2 = adj.copy(); adj2[ev_m] = adj[ev_m] + cmap2[be2]
    g = sc(adj2, ev_m) - sc(v88_final, ev_m)
    gains.append(g)
print(f'  둘다 적용 순수기여 평균 = {np.mean(gains):+.2f}  (H1->H2={gains[0]:+.2f} H2->H1={gains[1]:+.2f})')
