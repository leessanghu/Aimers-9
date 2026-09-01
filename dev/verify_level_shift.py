"""레벨 프로브 검증 (1) 시즌별 base rate 추세 (2) unc=0.249807의 정체 확인."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()

print('=== 시즌별 성공률(base rate) ===')
rates = {}
for s in sorted(pd.unique(season)):
    m = season == s
    r = y[m].mean()
    rates[s] = r
    print(f'  {int(s)}  n={m.sum():8,}  base_rate={r:.6f}   var=r(1-r)={r*(1-r):.6f}')

print()
UNC = 0.249807
print(f'대회 baseline 상수 unc = {UNC}')
disc = 0.25 - UNC
root = np.sqrt(max(disc, 0))
print(f'  만약 unc = p(1-p) 형태라면  p = {0.5+root:.6f}  또는  {0.5-root:.6f}')

print()
print('=== 학습 recency 가중 평균(프로덕션 half_life=2.0, 기준연도 2024) ===')
w = 0.5 ** ((2024 - season.astype(float)) / 2.0)
wmean = np.average(y, weights=w)
print(f'  recency 가중 성공률 = {wmean:.6f}')
print(f'  단순 전체평균       = {y.mean():.6f}')
print(f'  2024 단독           = {rates.get(2024, float("nan")):.6f}')

print()
print('=== 시즌별 추세(선형 외삽으로 2025 추정) ===')
ss = np.array(sorted(rates.keys()), dtype=float)
rr = np.array([rates[s] for s in sorted(rates.keys())])
coef = np.polyfit(ss, rr, 1)
pred2025 = np.polyval(coef, 2025.0)
print(f'  기울기 = {coef[0]:+.6f}/년')
print(f'  2025 선형외삽 추정 = {pred2025:.6f}')
recent = np.polyfit(ss[-3:], rr[-3:], 1)
print(f'  최근3년 기울기 = {recent[0]:+.6f}/년,  2025 추정 = {np.polyval(recent, 2025.0):.6f}')
