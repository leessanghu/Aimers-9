"""B: 레짐 구조 분리 테스트. CatBoost + early stopping(공정비교).
피처는 전 연도로 만든 것 그대로 쓰고, 트리 fit 대상만 바꾼다.
 - soft: recency 가중치 half_life 스윕 (1.0/1.5/2.0/3.0/균등)
 - hard: 학습 연도 윈도우 자르기 (2021-2023, 2022-2023)
fold A(train<=2023 -> val 2024)."""
import numpy as np, pandas as pd, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from catboost import CatBoostClassifier
t0=time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X=pd.read_parquet('dev/featcache_X.parquet')
meta=pd.read_parquet('dev/featcache_meta.parquet')
season=meta['season'].to_numpy(); y=meta['control_success'].to_numpy(np.float64)
row_num=meta['row_num'].to_numpy()
va=season==2024
Xv,yv=X.loc[va],y[va]
unc=0.249807
sc=lambda p: 1e5*(1-np.mean((np.clip(p,0,1)-yv)**2)/unc)

def run(tag, train_mask, half_life):
    Xt=X.loc[train_mask].reset_index(drop=True); yt=y[train_mask]
    s=season[train_mask].astype(float); rn=row_num[train_mask]
    w=np.ones(len(s)) if half_life is None else 0.5**((s.max()-s)/half_life)
    order=np.argsort(rn); n_es=int(len(Xt)*0.92)
    ti,ei=order[:n_es],order[n_es:]
    m=CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                         loss_function='Logloss', random_seed=42, verbose=False,
                         min_data_in_leaf=200, early_stopping_rounds=50)
    m.fit(Xt.iloc[ti], yt[ti], sample_weight=w[ti], eval_set=(Xt.iloc[ei], yt[ei]))
    p=m.predict_proba(Xv)[:,1]
    log(f'{tag:34s} n={train_mask.sum():>9,} best_iter={m.get_best_iteration():>4} BSS={sc(p):8.1f}')
    return sc(p)

all_tr = season<=2023
log('=== soft: recency half_life 스윕 (전체 연도 학습) ===')
base=None
for hl in [1.0, 1.5, 2.0, 3.0, None]:
    nm = '균등' if hl is None else f'half_life={hl}'
    v=run(f'soft {nm}', all_tr, hl)
    if hl==2.0: base=v
print()
log('=== hard: 학습 연도 윈도우 자르기 (half_life=2.0 유지) ===')
for yrs in [[2021,2022,2023],[2022,2023]]:
    run(f'hard {yrs[0]}-{yrs[-1]}', np.isin(season,yrs), 2.0)
print()
log(f'(기준 half_life=2.0 = {base:.1f})')
