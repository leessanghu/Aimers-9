"""idea75 — 11-class(구종) fold A 파일럿. v79의 mc5_model과 동일 구성으로 로컬 학습해
proba를 캐시하고, 디코더(E[y|c] vs 학습 선형)를 시간분할로 비교한다.

배경: 5-class에서 학습 선형 디코더가 시간분할 +162.14를 보였으나, v79에 실제 들어있는
모델은 11-class라 5-class 가중치를 직접 쓸 수 없다. 11-class proba가 로컬에 없어
여기서 생성한다. depth=5로 낮춰 시간을 줄인다(프로덕션 depth=6 대비 근사).
"""
import os, sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from catboost import CatBoostClassifier

CD="idea75_cache"; os.makedirs(CD,exist_ok=True); t0=time.time()
def log(m): print(f"[{time.time()-t0:5.0f}s] {m}",flush=True)

X=pd.read_parquet("featcache_X.parquet"); meta=pd.read_parquet("featcache_meta.parquet")
y=meta["control_success"].to_numpy(np.float64); season=meta["season"].to_numpy(np.float64)
cls5=np.load("cls5_labels.npy"); pt=np.load("pitchtype_labels.npy")
v=(cls5>=0)&(pt>=0)
cls=np.full(len(cls5),-1,dtype=np.int64)
nd=v&(cls5>=2); cls[nd]=(cls5[nd]-2)*3+pt[nd]
cls[v&(cls5==0)]=9; cls[v&(cls5==1)]=10
tr=season<=2023; va=season==2024
fit=tr&(cls>=0)
log(f"11-class 학습행 {fit.sum():,}  valid {va.sum():,}")
w=(0.5**((2023-season)/2.0))
fi=np.where(fit)[0]; n_es=int(len(fi)*0.92); ti,ei=fi[:n_es],fi[n_es:]

class CB:
    def __init__(s,p=20): s.p=p; s.t=time.time()
    def after_iteration(s,i):
        it=i.iteration; l=i.metrics["validation"]["MultiClass"][-1]
        if it%s.p==0: log(f"    iter {it:4d} loss={l:.6f} 경과={(time.time()-s.t)/60:.1f}분")
        return True

f=f"{CD}/A_proba11.npy"
if os.path.exists(f):
    P=np.load(f); log("캐시 사용")
else:
    m=CatBoostClassifier(iterations=400,learning_rate=0.05,depth=5,l2_leaf_reg=5.0,
        verbose=False,random_seed=42,loss_function="MultiClass",classes_count=11,
        early_stopping_rounds=40)
    m.fit(X.iloc[ti],cls[ti],sample_weight=w[ti],eval_set=(X.iloc[ei],cls[ei]),callbacks=[CB()])
    log(f"학습완료 best_iter={m.best_iteration_}")
    P=m.predict_proba(X.loc[va]); np.save(f,P)
log(f"proba shape={P.shape}")
np.save(f"{CD}/A_cls11_valid.npy", cls[va])
