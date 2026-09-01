import numpy as np, pandas as pd, joblib
df = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['row_id', 'season', 'pitcher_id', 'strikes_before'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)
df['is_2k'] = (df['strikes_before'] == 2).astype(int)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
unc = 0.249807

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
resid = yv - v88_final

tbl = pd.read_parquet('dev/pitcher_2k_slope_table.parquet')
va_idx = df.index[va]
pid_va = df.loc[va_idx, 'pitcher_id'].to_numpy()
is2k_va = df.loc[va_idx, 'is_2k'].to_numpy()
n2k_va = tbl['n2k'].reindex(pid_va).fillna(0).to_numpy(np.float64)
gap_va = tbl['gap'].reindex(pid_va).fillna(0).to_numpy(np.float64)
K = 1500
shrunk = gap_va * (n2k_va / (n2k_va + K))
applied = np.where(is2k_va == 1, shrunk, 0.0)
center = applied.mean()
cc = applied - center
C = np.mean(cc * resid)
V = np.mean(cc ** 2)
a = C / V
print(f'center={center:.5f}  C={C:.3e}  V={V:.3e}  alpha*={a:.4f}')
adj = v88_final + a * cc
sc = lambda q: 1e5 * (1 - np.mean((np.clip(q, 0, 1) - yv) ** 2) / unc)
print(f'v88_final={sc(v88_final):.2f}  +correction(alpha전체)={sc(adj):.2f}  (+{sc(adj)-sc(v88_final):.2f})')
adj_half = v88_final + 0.5 * a * cc
print(f'+correction(alpha*0.5, 안전판)={sc(adj_half):.2f}  (+{sc(adj_half)-sc(v88_final):.2f})')
