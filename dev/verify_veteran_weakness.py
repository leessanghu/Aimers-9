"""Q1(저표본) 1119 vs Q4(고표본) 786 이 진짜 모델 실력차인가, 고정 baseline 아티팩트인가.
층마다 라벨분산이 다르면 BSS가 실력과 무관하게 벌어진다.
-> 각 층에서 '그 층 자체의 분산' 대비 skill을 재야 진짜 비교다."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

B_FIX = 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
blend = np.load('dev/cache_v88_final_2024.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['season', 'asof_pitcher_n'])
n_ = np.nan_to_num(df.loc[va, 'asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
n_tr = np.nan_to_num(df.loc[season <= 2023, 'asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
qs = np.quantile(n_tr, [0.25, 0.5, 0.75])

strata = [
    ('Q1 저표본 (<%.0f)' % qs[0], n_ < qs[0]),
    ('Q2 (%.0f~%.0f)' % (qs[0], qs[1]), (n_ >= qs[0]) & (n_ < qs[1])),
    ('Q3 (%.0f~%.0f)' % (qs[1], qs[2]), (n_ >= qs[1]) & (n_ < qs[2])),
    ('Q4 고표본 (>=%.0f)' % qs[2], n_ >= qs[2]),
]

print(f'{"층":22s} {"n행":>8s} {"성공률":>8s} {"층분산":>9s} {"고정BSS":>9s} {"층자체BSS":>10s} {"corr(p,y)":>10s} {"예측std":>8s}')
for tag, m in strata:
    yy = yv[m]
    pp = blend[m]
    r = yy.mean()
    var_own = r * (1 - r)
    bs = np.mean((np.clip(pp, 0, 1) - yy) ** 2)
    bss_fix = 1e5 * (1 - bs / B_FIX)
    bss_own = 1e5 * (1 - bs / var_own)
    corr = np.corrcoef(pp, yy)[0, 1]
    print(f'{tag:22s} {m.sum():>8,} {r:>8.4f} {var_own:>9.6f} {bss_fix:>9.1f} {bss_own:>10.1f} '
          f'{corr:>+10.5f} {pp.std():>8.5f}')

print()
print('=== 해석 ===')
print('  고정BSS: 우리가 아까 본 수치 (층 성공률이 0.5에서 멀수록 자동으로 높아짐)')
print('  층자체BSS: 그 층 안에서의 진짜 모델 실력 (층 성공률 효과 제거)')
print('  corr(p,y): 예측-실제 상관. 이게 진짜 판별력.')

print()
print('=== 참고: 층별 예측평균 vs 실제평균 (레벨 정확도) ===')
for tag, m in strata:
    print(f'  {tag:22s} 예측평균={blend[m].mean():.4f}  실제={yv[m].mean():.4f}  편차={blend[m].mean()-yv[m].mean():+.5f}')
