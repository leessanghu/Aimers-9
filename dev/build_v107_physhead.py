"""v107 = v95 + 물리/커맨드 헤드(신규 블렌드 멤버).
- 신규 trackman 프로파일(tm_spin_sd/tm_velo_loss/tm_k2_rel_sd/tm_type_sep) 아티팩트 저장
- 물리헤드를 전체데이터(2019-2024)로 학습, multi-task [y, 1-middle, 1-reverse]
- 기존 10개 헤드 가중치를 전부 (1-W_NEW)배로 비례 축소해서 재원 마련
  (v99/v101 교훈: 특정 헤드를 0으로 만들면 실측 대손해)
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

W_NEW = 0.06
K_SD = 150.0
K_PROFILE = 200.0
NEWTM_COLS = ['tm_spin_sd', 'tm_velo_loss', 'tm_k2_rel_sd', 'tm_type_sep']

# ---------- 1) 신규 trackman 프로파일 ----------
log('trackman 로드...')
m_ = pd.read_csv('dev/pitcher_map.csv').sort_values('sim', ascending=False).drop_duplicates('tm_id')
t2p = m_.set_index('tm_id')['pitcher_id']
USECOLS = ['season', 'balls_before', 'strikes_before', 'pitcher_trackman_id', 'pitch_type_group',
           'rel_speed', 'spin_rate', 'rel_height', 'rel_side', 'zone_speed']
tm = pd.read_csv('data/trackman_history.csv', encoding='utf-8-sig', usecols=USECOLS)
tm = tm.rename(columns={'pitcher_trackman_id': 'tm_id'})
tm['pitcher_id'] = tm['tm_id'].map(t2p)
tm = tm.dropna(subset=['pitcher_id'])
tm['pitcher_id'] = tm['pitcher_id'].astype(np.int64)
log(f'  {len(tm):,}행')


def within_type_sd(tm, col, out):
    g = tm.groupby(['pitcher_id', 'season', 'pitch_type_group'])[col]
    cell = g.agg(['count', 'std']).reset_index()
    cell = cell[cell['count'] >= 2]
    gsd = float(cell['std'].median())
    cell['sd_sh'] = (cell['count'] * cell['std'].fillna(gsd) + K_SD * gsd) / (cell['count'] + K_SD)
    cell['wsum'] = cell['sd_sh'] * cell['count']
    agg = cell.groupby(['pitcher_id', 'season']).agg(wsum=('wsum', 'sum'), n=('count', 'sum'))
    return (agg['wsum'] / agg['n']).rename(out)


spin_sd = within_type_sd(tm, 'spin_rate', 'tm_spin_sd')

d = tm[['pitcher_id', 'season', 'rel_speed', 'zone_speed']].dropna()
d['loss'] = d['rel_speed'] - d['zone_speed']
velo_loss = d.groupby(['pitcher_id', 'season'])['loss'].mean().rename('tm_velo_loss')

d = tm[['pitcher_id', 'season', 'strikes_before', 'rel_height', 'rel_side']].dropna()
d['r2'] = np.sqrt(d['rel_height'] ** 2 + d['rel_side'] ** 2)
allsd = d.groupby(['pitcher_id', 'season'])['r2'].agg(['std', 'count'])
p2 = d[d['strikes_before'] >= 2]
psd = p2.groupby(['pitcher_id', 'season'])['r2'].agg(['std', 'count'])
j = allsd.join(psd, how='left', lsuffix='_all', rsuffix='_p')
gsd = float(j['std_all'].median())
kk = 80.0
psd_sh = (j['count_p'].fillna(0) * j['std_p'].fillna(gsd) + kk * j['std_all'].fillna(gsd)) / \
         (j['count_p'].fillna(0) + kk)
k2_rel_sd = (psd_sh - j['std_all'].fillna(gsd)).rename('tm_k2_rel_sd')

d = tm[['pitcher_id', 'season', 'pitch_type_group', 'rel_height', 'rel_side']].dropna()
cell = d.groupby(['pitcher_id', 'season', 'pitch_type_group']).agg(
    n=('rel_height', 'size'), h=('rel_height', 'mean'), s=('rel_side', 'mean')).reset_index()
cell = cell[cell['n'] >= 20]


def _sep(grp):
    if len(grp) < 2:
        return 0.0
    w_ = grp['n'].to_numpy(np.float64)
    hb = np.average(grp['h'], weights=w_)
    sb = np.average(grp['s'], weights=w_)
    var = np.average((grp['h'] - hb) ** 2 + (grp['s'] - sb) ** 2, weights=w_)
    return float(np.sqrt(var))


sep = cell.groupby(['pitcher_id', 'season']).apply(_sep, include_groups=False).rename('tm_type_sep')
ntypes = cell.groupby(['pitcher_id', 'season'])['n'].sum()
gsep = float(sep.median())
sep_sh = ((ntypes * sep + K_SD * gsep) / (ntypes + K_SD)).rename('tm_type_sep')

base_n = tm.groupby(['pitcher_id', 'season']).size().rename('tm_n')
prof = pd.concat([base_n, spin_sd, velo_loss, k2_rel_sd, sep_sh], axis=1).reset_index()
log(f'프로파일 {len(prof):,}')


def expanding(prof):
    rows = []
    for pid, grp in prof.groupby('pitcher_id'):
        grp = grp.sort_values('season')
        n_cum = 0.0
        acc = {c: 0.0 for c in NEWTM_COLS}
        for _, r in grp.iterrows():
            n = float(r['tm_n']) if np.isfinite(r['tm_n']) else 0.0
            for c in acc:
                v = r[c]
                if np.isfinite(v):
                    acc[c] += v * n
            n_cum += n
            out = {'pitcher_id': pid, 'season': int(r['season']), 'tm_n': n_cum}
            for c in acc:
                out[c] = acc[c] / n_cum if n_cum > 0 else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


exp = expanding(prof)
meta = pd.read_parquet('dev/featcache_meta.parquet')
seasons_range = list(range(int(exp['season'].min()), int(meta['season'].max()) + 2))
newtm_stats = {
    'profile': exp,
    'seasons_range': seasons_range,
    'k': K_PROFILE,
    'global_median': {c: float(exp[c].median()) for c in NEWTM_COLS},
}
log(f'newtm_stats 준비 (seasons_range={seasons_range[0]}~{seasons_range[-1]})')

# ---------- 2) 물리헤드 학습 (전체데이터) ----------
X = pd.read_parquet('dev/featcache_X.parquet')
newf = pd.read_parquet('dev/new_tm_features.parquet')
X = pd.concat([X.reset_index(drop=True), newf.reset_index(drop=True)], axis=1)
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
cls5 = np.load('dev/cls5_labels.npy')
valid5 = cls5 >= 0
h1 = np.where(valid5, 1.0 - (cls5 == 0).astype(np.float64), np.nan)
h2 = np.where(valid5, 1.0 - (cls5 == 1).astype(np.float64), np.nan)
Ymat = np.column_stack([y, h1, h2])

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
tm_feats = [c for c in v88['feature_order'] if c.startswith('tm_')]
NEWF = list(newf.columns)
CTX = [c for c in ['count_state', 'balls_before', 'strikes_before', 'outs_before', 'inning',
                   'pitcher_hand', 'batter_hand', 'same_hand', 'hand_matchup',
                   'x_count_pressure', 'asof_pitcher_n'] if c in X.columns]
FEATS = tm_feats + NEWF + CTX
log(f'물리헤드 피처 {len(FEATS)}개')

w = 0.5 ** ((2024.0 - season) / 2.0)
n = len(y)
n_es = int(n * 0.92)
CFG = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50)
log('물리헤드 학습(전체데이터)...')
mdl = CatBoostRegressor(**CFG, random_seed=42)
mdl.fit(X.iloc[:n_es][FEATS], Ymat[:n_es], sample_weight=w[:n_es],
        eval_set=(X.iloc[n_es:][FEATS], Ymat[n_es:]))
log(f'학습완료 best_iter={mdl.best_iteration_}')

_RNG = ('Generator', 'BitGenerator', 'RandomState')
def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, '__dict__'):
        for k2, v2 in list(vars(obj).items()):
            if type(v2).__name__ in _RNG:
                setattr(obj, k2, None)
            else:
                strip_rng(v2, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v2 in obj:
            strip_rng(v2, seen, depth + 1)
strip_rng(mdl)

# ---------- 3) v107 아티팩트 ----------
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v107 = dict(v95)
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
print('\n=== 가중치 재배분 (기존 전부 비례축소) ===')
for k2 in HEADS:
    old = float(v95[f'{k2}_weight'])
    new = old * (1 - W_NEW)
    v107[f'{k2}_weight'] = new
    print(f'  {k2:12s} {old:.4f} -> {new:.4f}')
v107['physhead_weight'] = W_NEW
v107['physhead_model'] = mdl
v107['physhead_feats'] = FEATS
v107['newtm_stats'] = newtm_stats
tot = sum(float(v107[f'{k2}_weight']) for k2 in HEADS) + W_NEW
print(f'  physhead     0.0000 -> {W_NEW:.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v107, 'submit/model/model_artifacts_v107.pkl')
log('v107 저장 완료')
