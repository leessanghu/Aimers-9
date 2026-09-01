"""LightGBM을 CatBoost와 동일한 정규화 사다리 3단계로 half_life=2.0 vs 균등 비교.
CatBoost 실험(exp_regularization.py)과 나란히 놓고 '모델 종류 문제냐 정규화 문제냐' 판별."""
import numpy as np, pandas as pd, time, sys
sys.stdout.reconfigure(encoding='utf-8')
import lightgbm as lgb
t0=time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X=pd.read_parquet('dev/featcache_X.parquet')
meta=pd.read_parquet('dev/featcache_meta.parquet')
season=meta['season'].to_numpy(); y=meta['control_success'].to_numpy(np.float64)
tr=season<=2023; va=season==2024
Xt=X.loc[tr].reset_index(drop=True); yt=y[tr]
Xv,yv=X.loc[va],y[va]
unc=0.249807
sc=lambda p: 1e5*(1-np.mean((np.clip(p,0,1)-yv)**2)/unc)
s_tr=season[tr].astype(float)

tr_meta = meta.loc[tr].reset_index(drop=True)
order = np.argsort(tr_meta['row_num'].to_numpy())
n_es = int(len(Xt)*0.92)
ti, ei = order[:n_es], order[n_es:]

CONFIGS = [
    ('프로덕션수준(l2=5,leaves=63,ff=1.0,minleaf=200)',
        dict(num_leaves=63, lambda_l2=5.0, feature_fraction=1.0, bagging_fraction=1.0, min_data_in_leaf=200)),
    ('중간(l2=20,leaves=63,ff=0.8,minleaf=500)',
        dict(num_leaves=63, lambda_l2=20.0, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, min_data_in_leaf=500)),
    ('강함(l2=50,leaves=63,ff=0.7,minleaf=2000, 원래실험설정)',
        dict(num_leaves=63, lambda_l2=50.0, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1, min_data_in_leaf=2000)),
]

for cname, extra in CONFIGS:
    row=[]
    for wname, w in [('half_life=2.0', 0.5**((2023-s_tr)/2.0)), ('균등', np.ones_like(s_tr))]:
        params = dict(objective='binary', metric='binary_logloss', learning_rate=0.05,
                      verbose=-1, num_threads=7)
        params.update(extra)
        dtr = lgb.Dataset(Xt.iloc[ti], yt[ti], weight=w[ti])
        dva = lgb.Dataset(Xt.iloc[ei], yt[ei], weight=w[ei], reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict(Xv, num_iteration=m.best_iteration)
        row.append(sc(p))
        log(f'  [{cname}] {wname:16s} best_iter={m.best_iteration}  BSS={sc(p):.1f}')
    log(f'  >>> [{cname}] 균등-현재 = {row[1]-row[0]:+.1f}')
    print()
