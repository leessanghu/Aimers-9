"""조건부 레벨 편차 스캔.
전역 shift로는 못 잡는 '그룹별 레벨 어긋남'이 어느 축에서 가장 큰가.
증분이득 = sum_g frac_g * (bias_g - bias_global)^2 * 400309   (정확한 식, 아티팩트 없음)

주의: fold A의 절대 편차는 fold A 고유 문제(모델이 못 본 레짐)로 부풀려져 있다.
여기서 보는 건 '어느 축에 조건부 구조가 있는가'이지 절대 이득 크기가 아니다."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

K = 1e5 / 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
blend = np.load('dev/cache_v88_final_2024.npy')
resid = blend - yv                     # 편차(예측-실제)
b_glob = resid.mean()
Xv = X.loc[va]
raw = pd.read_csv('data/train.csv', encoding='utf-8-sig')
raw = raw[raw['season'] == 2024].reset_index(drop=True)
print(f'전역 편차 = {b_glob:+.6f}   전역 shift 이득 = {b_glob**2*K:+.2f}\n')


def scan(name, groups, min_n=2000):
    vals = pd.Series(groups)
    inc = 0.0
    rows = []
    for gv, idx in vals.groupby(vals).groups.items():
        m = np.zeros(len(yv), bool)
        m[np.asarray(idx)] = True
        if m.sum() < min_n:
            continue
        bg = resid[m].mean()
        frac = m.sum() / len(yv)
        inc += frac * (bg - b_glob) ** 2
        rows.append((gv, m.sum(), bg))
    gain = inc * K
    print(f'{name:34s} 그룹={len(rows):3d}  증분이득={gain:+7.2f}')
    if gain > 1.5:
        rows.sort(key=lambda r: -abs(r[2] - b_glob))
        for gv, n, bg in rows[:6]:
            print(f'      {str(gv)[:26]:28s} n={n:>7,}  편차={bg:+.5f}  (전역대비 {bg-b_glob:+.5f})')
    return gain


n_ = np.nan_to_num(raw['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
n_tr = np.nan_to_num(pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['season', 'asof_pitcher_n']).query('season<=2023')['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)

res = {}
res['asof_pitcher_n 4분위'] = scan('asof_pitcher_n 4분위', np.digitize(n_, np.quantile(n_tr, [.25, .5, .75])))
res['asof_pitcher_n 10분위'] = scan('asof_pitcher_n 10분위', np.digitize(n_, np.quantile(n_tr, np.arange(.1, 1., .1))))
res['game_month'] = scan('game_month', raw['game_month'].to_numpy())
res['count_state'] = scan('count_state (볼-스트라이크)', Xv['count_state'].to_numpy())
res['inning'] = scan('inning', np.clip(raw['inning'].to_numpy(), 1, 10))
res['outs_before'] = scan('outs_before', raw['outs_before'].to_numpy())
res['same_hand'] = scan('same_hand', Xv['same_hand'].to_numpy())
res['pitcher_team'] = scan('pitcher_team_id', raw['pitcher_team_id'].to_numpy())
res['batter_team'] = scan('batter_team_id', raw['batter_team_id'].to_numpy())
res['base_state'] = scan('base_state', raw['base_state'].to_numpy())
res['num_runners'] = scan('num_runners_on', raw['num_runners_on'].to_numpy())
res['game_type'] = scan('game_type', raw['game_type'].to_numpy())
res['top_bottom'] = scan('top_bottom', raw['top_bottom'].to_numpy())
res['li 4분위'] = scan('li 4분위', np.digitize(raw['li'].to_numpy(), np.nanquantile(raw['li'], [.25, .5, .75])))
res['pred 10분위'] = scan('예측값 10분위', np.digitize(blend, np.quantile(blend, np.arange(.1, 1., .1))))
res['batter_n 4분위'] = scan('asof_batter_n 4분위', np.digitize(np.nan_to_num(raw['asof_batter_n'].to_numpy(np.float64), nan=0.0), [1000, 4000, 8000]))
res['fastball 4분위'] = scan('fastball_rate 4분위', np.digitize(np.nan_to_num(raw['asof_pitcher_fastball_rate'].to_numpy(np.float64), nan=0.5), [.4, .5, .6]))

print('\n=== 증분이득 순위 ===')
for k2, v2 in sorted(res.items(), key=lambda kv: -kv[1])[:8]:
    print(f'  {k2:28s} {v2:+7.2f}')
