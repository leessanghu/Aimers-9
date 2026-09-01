"""정규화 가설 검증: CatBoost 정규화를 세게 올리면 균등가중이 이기는가?
프로덕션(l2=5) vs 강한정규화(l2=50, rsm=0.5, depth 낮춤) x (half_life=2.0 vs 균등)"""
import numpy as np, pandas as pd, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from catboost import CatBoostClassifier
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
    ('프로덕션(l2=5,depth6,rsm=1.0)',  dict(depth=6, l2_leaf_reg=5.0)),
    ('중간(l2=20,rsm=0.8)',            dict(depth=6, l2_leaf_reg=20.0, rsm=0.8)),
    ('강함(l2=50,depth4,rsm=0.6)',     dict(depth=4, l2_leaf_reg=50.0, rsm=0.6)),
    ('LightGBM유사(l2=50,depth4,rsm=0.6,minleaf=2000)', dict(depth=4, l2_leaf_reg=50.0, rsm=0.6, min_data_in_leaf=2000)),
]
for cname, extra in CONFIGS:
    row=[]
    for wname, w in [('half_life=2.0', 0.5**((2023-s_tr)/2.0)), ('균등', np.ones_like(s_tr))]:
        params = dict(iterations=3000, learning_rate=0.03, loss_function='Logloss',
                      random_seed=42, verbose=False, min_data_in_leaf=200, early_stopping_rounds=50)
        params.update(extra)
        m=CatBoostClassifier(**params)
        m.fit(Xt.iloc[ti], yt[ti], sample_weight=w[ti], eval_set=(Xt.iloc[ei], yt[ei]))
        p=m.predict_proba(Xv)[:,1]
        row.append(sc(p))
        log(f'  [{cname}] {wname:16s} best_iter={m.get_best_iteration()}  BSS={sc(p):.1f}')
    log(f'  >>> [{cname}] 균등-현재 = {row[1]-row[0]:+.1f}')
    print()
