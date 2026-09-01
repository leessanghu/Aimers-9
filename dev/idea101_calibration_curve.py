"""v88_final(honest, fold A OOF: train<=2023 -> predict 2024)의 캘리브레이션 곡선을
최대한 잘게 쪼개서 확인. 예측분포가 0.5 근처에 몰려있는지, 그리고 어긋난 구간이
있으면 방향까지 확인."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

pred = np.load('dev/cache_v88_final_2024.npy')
df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y_all = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
va = season == 2024
y = y_all[va]
n = len(y)
log(f'fold A eval n={n:,}, 예측범위=[{pred.min():.4f}, {pred.max():.4f}]')

# 1) 예측분포 히스토그램 (0.5 근처 쏠림 확인)
edges = np.linspace(0, 1, 41)
hist, _ = np.histogram(pred, bins=edges)
print('\n=== 예측확률 히스토그램 (40구간) ===')
for i in range(40):
    lo, hi = edges[i], edges[i+1]
    bar = '#' * int(100 * hist[i] / n)
    print(f'  [{lo:.3f},{hi:.3f}) n={hist[i]:7,} ({100*hist[i]/n:5.2f}%) {bar}')
pct_mid = np.mean((pred >= 0.4) & (pred < 0.6))
pct_narrow = np.mean((pred >= 0.45) & (pred < 0.55))
print(f'\n[0.4,0.6) 비율 = {pct_mid*100:.2f}%   [0.45,0.55) 비율 = {pct_narrow*100:.2f}%')
print(f'예측 표준편차 = {pred.std():.4f}   실제 y 표준편차(참고) = {y.std():.4f}')

# 2) reliability curve - 최대한 잘게(등개수 구간), 통계적으로 유의한 어긋남만 표시
order = np.argsort(pred)
pred_s = pred[order]; y_s = y[order]

for K in (100, 300):
    print(f'\n=== Reliability curve: 등개수 {K}구간 (구간당 n≈{n//K:,}) ===')
    edges_idx = np.linspace(0, n, K + 1).astype(int)
    flagged = []
    for i in range(K):
        s, e = edges_idx[i], edges_idx[i+1]
        if e <= s:
            continue
        p_bin = pred_s[s:e]; y_bin = y_s[s:e]
        m_pred = p_bin.mean(); m_act = y_bin.mean(); cnt = e - s
        se = np.sqrt(max(m_act * (1 - m_act), 1e-9) / cnt)
        gap = m_act - m_pred
        z = gap / se if se > 0 else 0.0
        if abs(z) > 3:
            flagged.append((m_pred, m_act, cnt, gap, z))
    print(f'  |z|>3 (유의한 어긋남) 구간 수 = {len(flagged)} / {K}')
    for m_pred, m_act, cnt, gap, z in flagged[:30]:
        direction = '실제>예측(과소신)' if gap > 0 else '실제<예측(과신)'
        print(f'   pred={m_pred:.4f} actual={m_act:.4f} n={cnt:,} gap={gap:+.4f} z={z:+.1f} {direction}')

log('완료')
