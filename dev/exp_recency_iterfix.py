"""균등가중 vs half_life=2.0: iteration을 고정하지 않고 early_stopping으로 각자 최적점을 찾게 함.
CatBoost c1(depth=6,l2=5) 하나로 빠르게 확인. eval_set은 2023 내부 시간분할(마지막 2개월)로 구성."""
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

# eval_set: train 내부에서 시간순 마지막 8% (early_stopping용 홀드아웃, 2023 후반부 위주)
tr_meta = meta.loc[tr].reset_index(drop=True)
order = np.argsort(tr_meta['row_num'].to_numpy()) if 'row_num' in tr_meta.columns else np.arange(len(Xt))
n_es = int(len(Xt)*0.92)
ti, ei = order[:n_es], order[n_es:]
log(f'train={len(ti):,} es_holdout={len(ei):,}')

for wname, w in [('half_life=2.0', 0.5**((2023-s_tr)/2.0)), ('균등', np.ones_like(s_tr))]:
    m=CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                          loss_function='Logloss', random_seed=42, verbose=False,
                          min_data_in_leaf=200, early_stopping_rounds=50)
    m.fit(Xt.iloc[ti], yt[ti], sample_weight=w[ti], eval_set=(Xt.iloc[ei], yt[ei]))
    best_it = m.get_best_iteration()
    p_full = m.predict_proba(Xv)[:,1]
    log(f'{wname:16s} best_iter={best_it}  fold A(2024) BSS={sc(p_full):.1f}')
