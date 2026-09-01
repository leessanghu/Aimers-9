"""'피로도' 프록시: pitcher_id 연속블록(중간에 다른 투수 끼면 끊김) 안에서
몇 번째인지(position_in_stint) 계산. 뒤로 갈수록 성공확률이 떨어지는지 확인.
음수가중치(반대신호) 후보가 될 수 있는지 EB테이블+H1/H2로 검증."""
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
pos = df.groupby('block_id').cumcount().to_numpy() + 1  # 1부터 시작
block_len = df.groupby('block_id')['pitcher_id'].transform('size').to_numpy()
log(f'블록 수={block_id.max():,}  position 범위=[{pos.min()},{pos.max()}]')

# 1) 단순 상관: position이 클수록 성공률 낮아지나 (train<=2023 전체로)
tr = season <= 2023
order = np.argsort(pos[tr])
p_s = pos[tr][order]; y_s = y[tr][order]
K = 20
edges = np.linspace(0, len(p_s), K + 1).astype(int)
print('\n=== position_in_stint 20분위별 성공률 (train<=2023) ===')
for i in range(K):
    s, e = edges[i], edges[i+1]
    print(f'  pos [{p_s[s]:3d},{p_s[e-1]:3d}]  n={e-s:7,}  성공률={y_s[s:e].mean():.4f}')

va = season == 2024
yv = y[va]
X_va_pos = pos[va]

# 2) EB 축소 테이블(nbin=8, idea93 방식) + H1/H2
nbin = 8
tr_pos = pos[tr]; tr_y = y[tr]
qs = np.quantile(tr_pos, np.linspace(0, 1, nbin + 1))
qs[0] -= 1; qs[-1] += 1
bin_tr = np.digitize(tr_pos, qs) - 1
bin_va = np.digitize(X_va_pos, qs) - 1
global_mean = tr_y.mean()
K_SHRINK = 500
tbl = {}
for b in range(nbin):
    m = bin_tr == b
    n = m.sum(); mean_b = tr_y[m].mean() if n > 0 else global_mean
    shrunk = (n * mean_b + K_SHRINK * global_mean) / (n + K_SHRINK)
    tbl[b] = shrunk
sig = np.array([tbl.get(b, global_mean) for b in bin_va])

meta = pd.read_parquet('dev/featcache_meta.parquet')
X_ = pd.read_parquet('dev/featcache_X.parquet')
mth = X_.loc[meta['season'].to_numpy() == 2024, 'game_month'].to_numpy()
v88_final = np.load('dev/cache_v88_final_2024.npy')
sc = lambda p_, m: 1e5 * (1 - np.mean((np.clip(p_[m], 0, 1) - yv[m]) ** 2) / unc)

d = sig - sig.mean()
resid = yv - v88_final
H1 = mth <= 6; H2 = ~H1
print(f'\n=== position_in_stint EB테이블(nbin=8) 단독 상관 + H1/H2 additive 검증 ===')
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
