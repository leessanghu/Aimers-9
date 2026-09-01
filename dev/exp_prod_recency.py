"""프로덕션 학습기(HGB 3변종 + CatBoost 3변종)로 recency 가중치 효과 직접 측정.
fold A: train<=2023 -> val 2024.  half_life=2.0(현재) vs 균등."""
import numpy as np, pandas as pd, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
t0=time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X=pd.read_parquet('dev/featcache_X.parquet')
meta=pd.read_parquet('dev/featcache_meta.parquet')
season=meta['season'].to_numpy(); y=meta['control_success'].to_numpy(np.float64)
tr=season<=2023; va=season==2024
Xt,yt=X.loc[tr],y[tr]; Xv,yv=X.loc[va],y[va]
unc=0.249807
sc=lambda p: 1e5*(1-np.mean((np.clip(p,0,1)-yv)**2)/unc)
s_tr=season[tr].astype(float)

HGB=[('d6',dict(max_depth=6,max_leaf_nodes=31,random_state=42)),
     ('d8',dict(max_depth=8,max_leaf_nodes=15,random_state=2024)),
     ('sub',dict(max_depth=6,max_leaf_nodes=31,max_features=0.6,random_state=123))]
CAT=[('c1',dict(depth=6,l2_leaf_reg=5.0,random_seed=42)),
     ('c2',dict(depth=8,l2_leaf_reg=10.0,random_seed=7)),
     ('c3',dict(depth=6,l2_leaf_reg=5.0,rsm=0.6,random_seed=2024))]

res={}
for wname, w in [('half_life=2.0', 0.5**((2023-s_tr)/2.0)), ('균등', np.ones_like(s_tr))]:
    log(f'--- {wname} ---')
    ph=[]
    for n,ex in HGB:
        p=dict(max_iter=500,learning_rate=0.03,l2_regularization=5.0,early_stopping=False); p.update(ex)
        m=HistGradientBoostingClassifier(**p).fit(Xt,yt,sample_weight=w)
        pr=m.predict_proba(Xv)[:,1]; ph.append(pr); log(f'  hgb_{n}: {sc(pr):.1f}')
    pc=[]
    for n,ex in CAT:
        p=dict(iterations=1000,learning_rate=0.03,loss_function='Logloss',verbose=0,min_data_in_leaf=200); p.update(ex)
        m=CatBoostClassifier(**p).fit(Xt,yt,sample_weight=w)
        pr=m.predict_proba(Xv)[:,1]; pc.append(pr); log(f'  cat_{n}: {sc(pr):.1f}')
    base=0.5*np.mean(ph,axis=0)+0.5*np.mean(pc,axis=0)
    res[wname]=base
    log(f'  >>> base(HGB3+Cat3) = {sc(base):.1f}   예측평균={base.mean():.4f} (실제 {yv.mean():.4f})')

print()
print('='*60)
a,b=res['half_life=2.0'],res['균등']
print(f'현재(half_life=2.0) base : {sc(a):8.1f}')
print(f'균등가중 base            : {sc(b):8.1f}')
print(f'차이                     : {sc(b)-sc(a):+8.1f}')
np.save('dev/base_hl2_A.npy',a); np.save('dev/base_unif_A.npy',b)
