"""신규 trackman 파생피처 4개 + pitchmix 엔트로피 1개.
기존 trackman_profile.py 규약을 그대로 따름:
  - SD는 (투수,시즌,구종) 셀 내부에서 계산 후 K_SD 축소 -> 구종간 차이 제거
  - (pitcher_id, season) 프로파일 -> expanding 누적 -> season-1 룩업 -> K_PROFILE 축소

신규:
  tm_spin_sd    스핀 일관성 (기존엔 tm_spin_mean만 있고 SD 없음)
  tm_velo_loss  mean(rel_speed - zone_speed) = 종속 감속 (zone_speed 100% 미사용이었음)
  tm_k2_rel_sd  2스트라이크 릴리스 산포 - 평소 (press_rel_sd의 2K판, k2slope가 실측검증된 축)
  tm_type_sep   구종간 릴리스 평균의 분산 (_within_type_sd가 설계상 버린 성분)
  pitchmix_entropy  fb/br/os 비율의 엔트로피 (트리가 3변수 매끄러운 비선형함수를 근사 못함)
"""
import sys, os, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

TM_PATH = 'data/trackman_history.csv'
MAP_PATH = 'dev/pitcher_map.csv'
K_SD = 150.0
K_PROFILE = 200.0
NEW_COLS = ['tm_spin_sd', 'tm_velo_loss', 'tm_k2_rel_sd', 'tm_type_sep']

USECOLS = ['season', 'trackman_game_id', 'pitch_no', 'balls_before', 'strikes_before',
           'pitcher_trackman_id', 'pitch_type_group',
           'rel_speed', 'spin_rate', 'rel_height', 'rel_side', 'zone_speed']

log('trackman 로드...')
m = pd.read_csv(MAP_PATH).sort_values('sim', ascending=False).drop_duplicates('tm_id')
t2p = m.set_index('tm_id')['pitcher_id']
tm = pd.read_csv(TM_PATH, encoding='utf-8-sig', usecols=USECOLS)
tm = tm.rename(columns={'pitcher_trackman_id': 'tm_id'})
tm['pitcher_id'] = tm['tm_id'].map(t2p)
tm = tm.dropna(subset=['pitcher_id'])
tm['pitcher_id'] = tm['pitcher_id'].astype(np.int64)
log(f'  {len(tm):,}행, 투수 {tm.pitcher_id.nunique()}명')


def within_type_sd(tm, col, out):
    g = tm.groupby(['pitcher_id', 'season', 'pitch_type_group'])[col]
    cell = g.agg(['count', 'std']).reset_index()
    cell = cell[cell['count'] >= 2]
    gsd = float(cell['std'].median())
    cell['sd_sh'] = (cell['count'] * cell['std'].fillna(gsd) + K_SD * gsd) / (cell['count'] + K_SD)
    cell['wsum'] = cell['sd_sh'] * cell['count']
    agg = cell.groupby(['pitcher_id', 'season']).agg(wsum=('wsum', 'sum'), n=('count', 'sum'))
    return (agg['wsum'] / agg['n']).rename(out)


log('tm_spin_sd...')
spin_sd = within_type_sd(tm, 'spin_rate', 'tm_spin_sd')

log('tm_velo_loss...')
d = tm[['pitcher_id', 'season', 'rel_speed', 'zone_speed']].dropna()
d['loss'] = d['rel_speed'] - d['zone_speed']
velo_loss = d.groupby(['pitcher_id', 'season'])['loss'].mean().rename('tm_velo_loss')

log('tm_k2_rel_sd (2스트라이크 릴리스 산포 - 평소)...')
d = tm[['pitcher_id', 'season', 'strikes_before', 'rel_height', 'rel_side']].dropna()
d['r2'] = np.sqrt(d['rel_height'] ** 2 + d['rel_side'] ** 2)
allsd = d.groupby(['pitcher_id', 'season'])['r2'].agg(['std', 'count'])
p = d[d['strikes_before'] >= 2]
psd = p.groupby(['pitcher_id', 'season'])['r2'].agg(['std', 'count'])
j = allsd.join(psd, how='left', lsuffix='_all', rsuffix='_p')
gsd = float(j['std_all'].median())
k = 80.0
psd_sh = (j['count_p'].fillna(0) * j['std_p'].fillna(gsd) + k * j['std_all'].fillna(gsd)) / \
         (j['count_p'].fillna(0) + k)
k2_rel_sd = (psd_sh - j['std_all'].fillna(gsd)).rename('tm_k2_rel_sd')

log('tm_type_sep (구종간 릴리스 분리도)...')
d = tm[['pitcher_id', 'season', 'pitch_type_group', 'rel_height', 'rel_side']].dropna()
cell = d.groupby(['pitcher_id', 'season', 'pitch_type_group']).agg(
    n=('rel_height', 'size'), h=('rel_height', 'mean'), s=('rel_side', 'mean')).reset_index()
cell = cell[cell['n'] >= 20]


def _sep(grp):
    if len(grp) < 2:
        return 0.0
    w = grp['n'].to_numpy(np.float64)
    hb = np.average(grp['h'], weights=w)
    sb = np.average(grp['s'], weights=w)
    var = np.average((grp['h'] - hb) ** 2 + (grp['s'] - sb) ** 2, weights=w)
    return float(np.sqrt(var))


sep = cell.groupby(['pitcher_id', 'season']).apply(_sep, include_groups=False).rename('tm_type_sep')
ntypes = cell.groupby(['pitcher_id', 'season'])['n'].sum().rename('_nt')
gsep = float(sep.median())
sep_sh = ((ntypes * sep + K_SD * gsep) / (ntypes + K_SD)).rename('tm_type_sep')

base_n = tm.groupby(['pitcher_id', 'season']).size().rename('tm_n')
prof = pd.concat([base_n, spin_sd, velo_loss, k2_rel_sd, sep_sh], axis=1).reset_index()
log(f'프로파일 {len(prof):,}개 (투수x시즌)')
print(prof[NEW_COLS].describe().T)


def expanding(prof):
    rows = []
    for pid, grp in prof.groupby('pitcher_id'):
        grp = grp.sort_values('season')
        n_cum = 0.0
        acc = {c: 0.0 for c in NEW_COLS}
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


log('expanding 누적...')
exp = expanding(prof)

log('메인 데이터에 룩업(season-1) 적용...')
meta = pd.read_parquet('dev/featcache_meta.parquet')
df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['pitcher_id', 'season'])
seasons_range = list(range(int(exp['season'].min()), int(meta['season'].max()) + 1))
idx = pd.MultiIndex.from_arrays([df['pitcher_id'], df['season'] - 1])
glob = {c: float(exp[c].median()) for c in NEW_COLS}

piv_n = exp.pivot_table(index='pitcher_id', columns='season', values='tm_n', aggfunc='first')
piv_n = piv_n.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
n_cell = np.nan_to_num(piv_n.reindex(idx).to_numpy().astype(np.float64), nan=0.0)

out = pd.DataFrame(index=df.index)
for c in NEW_COLS:
    p_ = exp.pivot_table(index='pitcher_id', columns='season', values=c, aggfunc='first')
    p_ = p_.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    v = p_.reindex(idx).to_numpy().astype(np.float64)
    gm = glob[c]
    v = np.where(np.isfinite(v), v, gm)
    out[c] = (n_cell * v + K_PROFILE * gm) / (n_cell + K_PROFILE)

log('pitchmix 엔트로피...')
X = pd.read_parquet('dev/featcache_X.parquet',
                    columns=['asof_pitcher_fastball_rate_smooth',
                             'asof_pitcher_breaking_rate_smooth',
                             'asof_pitcher_offspeed_rate_smooth'])
P = X.to_numpy(np.float64)
P = np.clip(P, 1e-9, None)
P = P / P.sum(axis=1, keepdims=True)
out['pitchmix_entropy'] = -(P * np.log(P)).sum(axis=1)
out['pitchmix_maxshare'] = P.max(axis=1)

out = out.astype(np.float64)
out.to_parquet('dev/new_tm_features.parquet')
log(f'저장 완료: dev/new_tm_features.parquet  {out.shape}')
print(out.describe().T)
