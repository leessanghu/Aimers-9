import numpy as np, pandas as pd, lightgbm as lgb, time, sys
sys.stdout.reconfigure(encoding='utf-8')
t0=time.time()
X=pd.read_parquet('dev/featcache_X.parquet')
meta=pd.read_parquet('dev/featcache_meta.parquet')
season=meta['season'].to_numpy(); y=meta['control_success'].to_numpy(np.float64)
unc=0.249807
PARAMS=dict(objective='binary',metric='binary_logloss',learning_rate=0.05,
            num_leaves=63,min_data_in_leaf=200,feature_fraction=0.8,
            bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,verbose=-1,num_threads=7)
ROUNDS=500

def run(train_max, val_season):
    tr=season<=train_max; va=season==val_season
    Xv=X.loc[va]; yv=y[va]; ybar=yv.mean()
    s_tr=season[tr].astype(float)
    print(f'\n### fold: train<={train_max} -> val {val_season}  (실제성공률 {ybar:.4f}, n_tr={tr.sum():,})')
    print(f'{"recency":18s} {"예측평균":>9s} {"편차":>8s} {"BSS원본":>9s} {"BSS레벨보정":>11s}')
    print('-'*62)
    for name,hl in [('half_life=2.0',2.0),('half_life=4.0',4.0),('half_life=8.0',8.0),('균등',None)]:
        w = np.ones_like(s_tr) if hl is None else 0.5**((train_max-s_tr)/hl)
        m=lgb.train(PARAMS, lgb.Dataset(X.loc[tr], y[tr], weight=w), num_boost_round=ROUNDS)
        p=m.predict(Xv)
        raw=1e5*(1-np.mean((np.clip(p,0,1)-yv)**2)/unc)
        pc=np.clip(p-(p.mean()-ybar),0,1)   # 레벨을 실제 평균에 맞춤(오라클 레벨)
        cor=1e5*(1-np.mean((pc-yv)**2)/unc)
        print(f'{name:18s} {p.mean():9.4f} {p.mean()-ybar:+8.4f} {raw:9.1f} {cor:11.1f}   ({time.time()-t0:.0f}s)')

run(2023, 2024)
run(2021, 2022)
