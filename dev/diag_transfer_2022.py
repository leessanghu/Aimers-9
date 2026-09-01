"""2022년 구종재분류(trackman)가 control_success 라벨(train.csv)에 얼마나 전이됐나.
투수 단위 준실험: (2021->2022) 패스트볼 비중이 많이 바뀐 투수일수록
reverse/middle도 더 많이 바뀌었는지 상관을 본다.
전이됐다면(라벨 정의가 실제로 흔들림) 강한 상관, 안 됐다면(독립적 진짜 변화) 약한 상관.
"""
import numpy as np, pandas as pd, sys, time
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
def log(m): print(f'[{time.time()-t0:5.0f}s] {m}', flush=True)

pmap = pd.read_csv('dev/pitcher_map.csv')  # pitcher_id, tm_id, sim
pmap = pmap[pmap['sim'] >= 0.8].drop_duplicates('pitcher_id')
log(f'매핑 {len(pmap):,}명 (sim>=0.8)')

tm = pd.read_csv('data/trackman_history.csv', encoding='utf-8-sig',
                 usecols=['season', 'pitcher_trackman_id', 'pitch_type_group'])
tm_mix = tm.groupby(['pitcher_trackman_id', 'season'])['pitch_type_group'].apply(
    lambda s: (s == 'fastball').mean()).reset_index(name='fb_share')
log(f'trackman 구종비중 계산 {len(tm_mix):,}행')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'season', 'pitcher_id', 'asof_pitcher_n',
                          'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate',
                          'control_success'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)

# 행별 결과유형 복원 (이전과 동일 로직)
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
def cnt(col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
R_ = cnt('asof_pitcher_reverse_rate'); M_ = cnt('asof_pitcher_middle_rate')
ordr = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
pid_o = df['pitcher_id'].to_numpy()[ordr]
n_o = n_[ordr]
hstep = np.zeros(len(df), dtype=bool)
hstep[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
rev_d = np.zeros(len(df)); rev_d[ordr[:-1]] = np.diff(R_[ordr])
mid_d = np.zeros(len(df)); mid_d[ordr[:-1]] = np.diff(M_[ordr])
df['rev'] = np.where(hstep, (rev_d > 0).astype(float), np.nan)
df['mid'] = np.where(hstep, (mid_d > 0).astype(float), np.nan)
log(f'라벨 복원 완료 (유효 {hstep.sum():,}행)')

# 투수x시즌 라벨 집계
lab = df.groupby(['pitcher_id', 'season']).agg(
    succ=('control_success', 'mean'), rev=('rev', 'mean'), mid=('mid', 'mean'),
    n=('control_success', 'size')).reset_index()
lab = lab.merge(pmap[['pitcher_id', 'tm_id']], on='pitcher_id', how='left')
lab = lab.merge(tm_mix.rename(columns={'pitcher_trackman_id': 'tm_id'}), on=['tm_id', 'season'], how='left')

print('=== 매칭 확인 ===')
print(f'lab 전체 {len(lab):,}, fb_share 있는 것 {lab.fb_share.notna().sum():,}')
print()

for y0, y1 in [(2021, 2022), (2022, 2023), (2023, 2024)]:
    a = lab[(lab.season == y0) & (lab.n >= 150)].set_index('pitcher_id')
    b = lab[(lab.season == y1) & (lab.n >= 150)].set_index('pitcher_id')
    common = a.index.intersection(b.index)
    common = common[a.loc[common, 'fb_share'].notna() & b.loc[common, 'fb_share'].notna()]
    if len(common) < 20:
        print(f'{y0}->{y1}: 표본부족 n={len(common)}'); continue
    d_fb = b.loc[common, 'fb_share'] - a.loc[common, 'fb_share']
    d_rev = b.loc[common, 'rev'] - a.loc[common, 'rev']
    d_mid = b.loc[common, 'mid'] - a.loc[common, 'mid']
    d_succ = b.loc[common, 'succ'] - a.loc[common, 'succ']
    print(f'=== {y0}->{y1} (n={len(common)}명) ===')
    print(f'  패스트볼비중 변화: 평균={d_fb.mean():+.4f}  표준편차={d_fb.std():.4f}')
    print(f'  corr(d_fb, d_rev)  = {np.corrcoef(d_fb, d_rev)[0,1]:+.4f}')
    print(f'  corr(d_fb, d_mid)  = {np.corrcoef(d_fb, d_mid)[0,1]:+.4f}')
    print(f'  corr(d_fb, d_succ) = {np.corrcoef(d_fb, d_succ)[0,1]:+.4f}')
    # 패스트볼 비중 많이 줄어든 상위/하위 그룹 비교
    q = pd.qcut(d_fb, 3, labels=['많이감소', '중간', '증가/유지'])
    print(f'  그룹별 d_succ 평균:')
    print(d_succ.groupby(q, observed=True).mean().round(4).to_string())
    print()
log('완료')
