import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()

for flag in ['tm_matched', 'tm_lown_flag']:
    f = X[flag].to_numpy()
    print(f'=== {flag} vs 상위피처 교차표 ===')
    for top in ['cat_game_type', 'same_hand']:
        if top not in X.columns:
            continue
        t = X[top]
        ct = pd.crosstab(f, t, normalize='columns')
        print(f'  --- {top} 별 {flag}=1 비율 ---')
        print(ct.to_string())
    print()

print('=== season별: tm_matched=1비율, 성공률 ===')
tmm = X['tm_matched'].to_numpy()
for s in sorted(pd.Series(season).unique()):
    m = season == s
    print(f'  season={s}  tm_matched비율={tmm[m].mean():.3f}  성공률={y[m].mean():.4f}  n={m.sum():,}')

print()
print('=== tm_matched 조건부 game_type별 성공률 (진짜 독립신호인지) ===')
gt = X['cat_game_type'].to_numpy()
for g in sorted(pd.Series(gt).dropna().unique()):
    for tv in (0.0, 1.0):
        m = (gt == g) & (tmm == tv)
        if m.sum() < 100:
            continue
        print(f'  game_type={g} tm_matched={tv}  n={m.sum():>9,}  성공률={y[m].mean():.4f}')
