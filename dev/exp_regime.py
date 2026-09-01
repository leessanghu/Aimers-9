import numpy as np, pandas as pd, lightgbm as lgb, time, sys
sys.stdout.reconfigure(encoding='utf-8')
t0=time.time()
X=pd.read_parquet('dev/featcache_X.parquet')
meta=pd.read_parquet('dev/featcache_meta.parquet')
season=meta['season'].to_numpy(); y=meta['control_success'].to_numpy(np.float64)
va=season==2024; yv=y[va]; Xv=X.loc[va]
unc=0.249807
def bss(p): return 1e5*(1-np.mean((np.clip(p,0,1)-yv)**2)/unc)

# 2024를 월로 반 나눠 안정성 체크
mth=X.loc[va,'game_month'].to_numpy()
h1=mth<=6; h2=~h1
def bss_m(p,m): 
    yy=yv[m]; return 1e5*(1-np.mean((np.clip(p[m],0,1)-yy)**2)/unc)

PARAMS=dict(objective='binary',metric='binary_logloss',learning_rate=0.05,
            num_leaves=63,min_data_in_leaf=200,feature_fraction=0.8,
            bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=7)
ROUNDS=500

configs=[
 ('전체 half_life=2.0 (현재)', lambda s: 0.5**((2023-s)/2.0), None),
 ('전체 half_life=1.0',        lambda s: 0.5**((2023-s)/1.0), None),
 ('전체 half_life=0.5',        lambda s: 0.5**((2023-s)/0.5), None),
 ('전체 균등가중',              lambda s: np.ones_like(s,dtype=float), None),
 ('2023만',                    lambda s: np.ones_like(s,dtype=float), [2023]),
 ('2022-2023',                 lambda s: np.ones_like(s,dtype=float), [2022,2023]),
 ('2021-2023',                 lambda s: np.ones_like(s,dtype=float), [2021,2022,2023]),
]
print(f'val 2024 n={va.sum():,}  (H1={h1.sum():,} H2={h2.sum():,})')
print(f'{"config":28s} {"BSS(2024)":>10s} {"H1":>8s} {"H2":>8s} {"n_train":>10s}')
print('-'*70)
for name,wf,years in configs:
    tr = season<=2023 if years is None else np.isin(season,years)
    s_tr=season[tr].astype(float)
    w=wf(s_tr)
    d=lgb.Dataset(X.loc[tr], y[tr], weight=w)
    m=lgb.train(PARAMS,d,num_boost_round=ROUNDS)
    p=m.predict(Xv)
    print(f'{name:28s} {bss(p):10.1f} {bss_m(p,h1):8.1f} {bss_m(p,h2):8.1f} {tr.sum():10,}  ({time.time()-t0:.0f}s)')
