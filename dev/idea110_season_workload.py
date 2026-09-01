"""시즌 내 누적 등판횟수(워크로드) 신규 피처. asof_pitcher_n(커리어누적)과 달리
'이번 시즌 지금까지 몇 번째 등판인지'만 잡음. 많을수록 성공률이 떨어지는지 EB테이블+H1/H2."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
unc = 0.249807

same_prev = (df['pitcher_id'] == df['pitcher_id'].shift(1)).to_numpy()
block_id = (~same_prev).cumsum()
df['block_id'] = block_id
block_start = df.groupby('block_id').head(1).index
is_block_start = np.zeros(len(df), dtype=bool)
is_block_start[block_start] = True

# 시즌 내 등판순번: (season, pitcher_id) 그룹에서 block_start가 몇 번째인지 누적
df['is_start'] = is_block_start.astype(int)
df['season_appear_no'] = df.groupby(['season', 'pitcher_id'])['is_start'].cumsum()
appear_no = df['season_appear_no'].to_numpy()
log(f'등판순번 범위=[{appear_no.min()},{appear_no.max()}]  중앙값={np.median(appear_no):.0f}')

tr = season <= 2023; va = season == 2024
yv = y[va]

# 1) 단순 20분위 상관 확인 (train<=2023)
order = np.argsort(appear_no[tr])
a_s = appear_no[tr][order]; y_s = y[tr][order]
K = 20
edges = np.linspace(0, len(a_s), K + 1).astype(int)
print('\n=== season_appear_no 20분위별 성공률 (train<=2023) ===')
for i in range(K):
    s, e = edges[i], edges[i+1]
    print(f'  appear_no [{a_s[s]:4.0f},{a_s[e-1]:4.0f}]  n={e-s:7,}  성공률={y_s[s:e].mean():.4f}')

# 2) EB테이블(nbin=8) + H1/H2
nbin = 8
tr_a = appear_no[tr]; tr_y = y[tr]
qs = np.quantile(tr_a, np.linspace(0, 1, nbin + 1))
qs[0] -= 1; qs[-1] += 1
bin_tr = np.digitize(tr_a, qs) - 1
bin_va = np.digitize(appear_no[va], qs) - 1
global_mean = tr_y.mean()
K_SHRINK = 500
tbl = {}
for b in range(nbin):
    m = bin_tr == b
    n = m.sum(); mean_b = tr_y[m].mean() if n > 0 else global_mean
    tbl[b] = (n * mean_b + K_SHRINK * global_mean) / (n + K_SHRINK)
sig = np.array([tbl.get(b, global_mean) for b in bin_va])

meta = pd.read_parquet('dev/featcache_meta.parquet')
X_ = pd.read_parquet('dev/featcache_X.parquet')
mth = X_.loc[meta['season'].to_numpy() == 2024, 'game_month'].to_numpy()
v88_final = np.load('dev/cache_v88_final_2024.npy')
sc = lambda p_, m: 1e5 * (1 - np.mean((np.clip(p_[m], 0, 1) - yv[m]) ** 2) / unc)

d = sig - sig.mean()
resid = yv - v88_final
H1 = mth <= 6; H2 = ~H1
print(f'\n=== season_appear_no EB테이블(nbin=8) H1/H2 additive 검증 ===')
gains = []
for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
    C = np.mean(d[fit_m] * resid[fit_m]); V = np.mean(d[fit_m] ** 2)
    w_star = C / V if V > 1e-12 else 0.0
    blend = v88_final.copy(); blend[ev_m] = v88_final[ev_m] + w_star * d[ev_m]
    g = sc(blend, ev_m) - sc(v88_final, ev_m)
    gains.append(g)
    print(f'{tag}: w*={w_star:+.4f}  이득={g:+.2f}')
print(f'평균 이득 = {np.mean(gains):+.2f}')
log('완료')
