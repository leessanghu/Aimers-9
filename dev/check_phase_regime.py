"""코덱스 phase 구분(조기3-4월/중기5-7월/후기8월+)으로 fold A 잔차를 쪼개서
1) 레벨(잔차평균)이 phase별로 다른가 2) 주요피처-잔차 관계가 phase별로 다른가."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

B = 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
va = season == 2024
yv = y[va]
base = np.load('dev/cache_v88_final_2024.npy')
resid = yv - base

mth = X.loc[va, 'game_month'].to_numpy()
phase = np.where(mth <= 4, 'early(3-4)', np.where(mth <= 7, 'mid(5-7)', 'late(8+)'))
sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)

print('=== (1) phase별 잔차 레벨 ===')
for ph in ['early(3-4)', 'mid(5-7)', 'late(8+)']:
    m = phase == ph
    print(f'  {ph:12s} n={m.sum():7,}  잔차평균={resid[m].mean():+.5f}  BSS={sc(base,m):.2f}')

print()
print('=== (2) 주요 피처 - 잔차 상관, phase별 ===')
CHECK = ['x_ability_here', 'inseason_cmd_index', 'x_count_pressure', 'strikes_before',
         'same_hand', 'asof_pitcher_offspeed_rate_smooth', 'li', 'season']
Xv = X.loc[va]
print(f'{"피처":32s} {"early":>10s} {"mid":>10s} {"late":>10s}')
for f in CHECK:
    if f not in Xv.columns:
        continue
    vals = Xv[f].to_numpy(np.float64)
    row = []
    for ph in ['early(3-4)', 'mid(5-7)', 'late(8+)']:
        m = phase == ph
        c = np.corrcoef(vals[m], resid[m])[0, 1]
        row.append(c)
    print(f'{f:32s} {row[0]:>+10.4f} {row[1]:>+10.4f} {row[2]:>+10.4f}')

print()
print('=== (3) phase 더미 자체가 잔차와 얼마나 관계있나 (최대이득 공식) ===')
K = 1e5 / B
for ph in ['early(3-4)', 'mid(5-7)', 'late(8+)']:
    g = (phase == ph).astype(np.float64)
    g = g - g.mean()
    C = np.mean(g * resid)
    V = np.mean(g ** 2)
    maxgain = (C * C / V) * K
    rho = C / np.sqrt(V * resid.var())
    print(f'  {ph:12s} corr={rho:+.5f}  이론최대이득={maxgain:+.2f}')
