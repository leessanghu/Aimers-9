"""(1) fold A에서 v88_final의 실제 편차(예측평균 - 실제) 측정
(2) 트리모델이 season=2025를 2024와 동일하게 취급하는지 직접 확인"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
va = season == 2024
yv = y[va]
UNC = 0.249807

pred = np.load('dev/cache_v88_final_2024.npy')
D = pred.mean() - yv.mean()
print('=== (1) fold A(train<=2023 -> 2024) 실제 레벨편차 ===')
print(f'  예측평균 = {pred.mean():.6f}')
print(f'  실제평균 = {yv.mean():.6f}')
print(f'  편차 D   = {D:+.6f}')
bs0 = np.mean((pred - yv) ** 2)
bs_shift = np.mean((pred - D - yv) ** 2)
sc = lambda b: 1e5 * (1 - b / UNC)
print(f'  보정전 BSS = {sc(bs0):.2f}')
print(f'  최적 전역보정({-D:+.6f}) 적용시 = {sc(bs_shift):.2f}   이득={sc(bs_shift)-sc(bs0):+.2f}')
print(f'  이론이득 400309*D^2 = {(1e5/UNC)*D*D:+.2f}')

print()
print('=== (2) 트리모델의 season 외삽 여부 (season=2024 vs 2025) ===')
X = pd.read_parquet('dev/featcache_X.parquet')
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
samp = np.where(season == 2024)[0][:20000]
Xs = X.iloc[samp].copy()

mc5 = v88['mc5_model']
p24 = mc5.predict_proba(Xs)
Xs2 = Xs.copy()
Xs2['season'] = 2025
p25 = mc5.predict_proba(Xs2)
print(f'  mc5 proba 최대절대차(2024 vs 2025) = {np.abs(p24 - p25).max():.10f}')

cat0 = v88['cats'][0]
c24 = cat0.predict_proba(Xs)[:, 1]
c25 = cat0.predict_proba(Xs2)[:, 1]
print(f'  base cat[0] 최대절대차            = {np.abs(c24 - c25).max():.10f}')
print(f'  base cat[0] 평균(2024)={c24.mean():.6f}  평균(2025)={c25.mean():.6f}')

hgb0 = v88['hgbs'][0]
h24 = hgb0.predict_proba(Xs)[:, 1]
h25 = hgb0.predict_proba(Xs2)[:, 1]
print(f'  base hgb[0] 최대절대차            = {np.abs(h24 - h25).max():.10f}')

print()
print('  -> 차이가 0이면 모델은 2025를 2024와 동일 취급(외삽 불가) 확정')
