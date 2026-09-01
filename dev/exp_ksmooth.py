"""K_SMOOTH 스윕: 시즌내 성공률 축소강도가 투수-시즌 실력 추정에 미치는 영향.
대회 구조상 이게 순위를 결정하는 축이다(투수-시즌 오라클이 실질 상한).
프로덕션은 K=15인데 주석상 이론최적 87 / forward검증최적 150.

1단계(무료, 노이즈0): 단변량 예측력 - K별 inseason_success_smooth 단독으로 y를 얼마나 설명하나
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

art = joblib.load('submit/model/model_artifacts_v88.pkl')
ins = art['inseason_stats']
se = ins['season_end_table']
g = ins['global_success_rate']
sr = ins['seasons_range']

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'season', 'pitcher_id', 'asof_pitcher_n',
                          'asof_pitcher_success_rate', 'control_success'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)

# 직전 시즌 끝 시점 누적 (script.py build_inseason_features와 동일 로직)
piv = {}
for col in ['N_end', 'S_end']:
    p = se.pivot(index='pitcher_id', columns='season', values=col)
    p = p.reindex(columns=sr).ffill(axis=1)
    piv[col] = p.stack(future_stack=True)
p = se.pivot(index='pitcher_id', columns='season', values='prior_success_rate')
p = p.reindex(columns=sr).ffill(axis=1)
piv['rate'] = p.stack(future_stack=True)

idx = pd.MultiIndex.from_arrays([df['pitcher_id'], df['season'] - 1])
N_end = np.nan_to_num(piv['N_end'].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
S_end = np.nan_to_num(piv['S_end'].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
prior = pd.Series(piv['rate'].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)

n_now = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
s_now = np.round(df['asof_pitcher_success_rate'].fillna(0).to_numpy(np.float64) * n_now)
n_season = np.clip(n_now - N_end, 0, None)
s_season = np.clip(s_now - S_end, 0, None)
rate_raw = np.divide(s_season, n_season, out=np.full_like(n_season, np.nan), where=n_season > 0)

season = df['season'].to_numpy()
y = df['control_success'].to_numpy(np.float64)
unc = 0.249807

print(f'n_season 분포: median={np.median(n_season):.0f}  mean={n_season.mean():.0f}')
print(f'  n_season<15인 행 비율: {(n_season<15).mean()*100:.2f}%')
print(f'  n_season<150인 행 비율: {(n_season<150).mean()*100:.2f}%')
print()

KS = [15, 30, 60, 87, 150, 250, 400, 700]
print(f'{"K":>5s} | ' + ' | '.join(f'{"corr(y)":>8s} {"단변량BSS":>10s}  [{v}]' for v in (2022, 2024)))
print('-' * 70)
for K in KS:
    smooth = (n_season * np.nan_to_num(rate_raw) + K * prior) / (n_season + K)
    row = []
    for vs in (2022, 2024):
        m = season == vs
        yv = y[m]; f = smooth[m]
        r = np.corrcoef(f, yv)[0, 1]
        # 단변량 예측: 전년도(vs-1) 데이터로 구간->평균y 맵 만들고 vs에 적용 (정직)
        mtr = season == (vs - 1)
        ftr = smooth[mtr]; ytr = y[mtr]
        edges = np.unique(np.quantile(ftr, np.linspace(0, 1, 51)))
        edges[0] -= 1e-9; edges[-1] += 1e-9
        bi_tr = np.clip(np.digitize(ftr, edges) - 1, 0, len(edges) - 2)
        bi_v = np.clip(np.digitize(f, edges) - 1, 0, len(edges) - 2)
        means = np.full(len(edges) - 1, ytr.mean())
        for b in range(len(edges) - 1):
            mm = bi_tr == b
            if mm.sum() >= 200:
                means[b] = ytr[mm].mean()
        pred = means[bi_v]
        bss = 1e5 * (1 - np.mean((pred - yv) ** 2) / unc)
        row.append(f'{r:+8.4f} {bss:10.1f}')
    star = '  <- 프로덕션' if K == 15 else ''
    print(f'{K:5d} | ' + ' | '.join(row) + star)
