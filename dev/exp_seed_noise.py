"""노이즈 바닥 측정: 같은 설정에서 seed만 바꿔 fold A BSS가 얼마나 흔들리는가?
이걸 모르면 half_life=3.0의 +18.2가 진짜인지 우연인지 판별 불가."""
import numpy as np, pandas as pd, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from catboost import CatBoostClassifier
t0=time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X=pd.read_parquet('dev/featcache_X.parquet')
meta=pd.read_parquet('dev/featcache_meta.parquet')
season=meta['season'].to_numpy(); y=meta['control_success'].to_numpy(np.float64)
row_num=meta['row_num'].to_numpy()
tr=season<=2023; va=season==2024
Xt=X.loc[tr].reset_index(drop=True); yt=y[tr]
Xv,yv=X.loc[va],y[va]
s=season[tr].astype(float); rn=row_num[tr]
order=np.argsort(rn); n_es=int(len(Xt)*0.92); ti,ei=order[:n_es],order[n_es:]
unc=0.249807
sc=lambda p: 1e5*(1-np.mean((np.clip(p,0,1)-yv)**2)/unc)

def fit(hl, seed):
    w=0.5**((s.max()-s)/hl)
    m=CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                         loss_function='Logloss', random_seed=seed, verbose=False,
                         min_data_in_leaf=200, early_stopping_rounds=50)
    m.fit(Xt.iloc[ti], yt[ti], sample_weight=w[ti], eval_set=(Xt.iloc[ei], yt[ei]))
    v=sc(m.predict_proba(Xv)[:,1])
    log(f'  half_life={hl}  seed={seed:4d}  best_iter={m.get_best_iteration():>4}  BSS={v:8.1f}')
    return v

for hl in [2.0, 3.0]:
    vals=[fit(hl,sd) for sd in (42,7,2024,123)]
    a=np.array(vals)
    log(f'>>> half_life={hl}: 평균={a.mean():.1f} 표준편차={a.std(ddof=1):.1f} 범위=[{a.min():.1f}, {a.max():.1f}]')
    print()
