"""max_iter 상한 문제 분리 + 용량 축소 방향 탐색 (baseline 67피처 고정).

phase15에서 baseline이 n_iter_=500 = max_iter로 끝남 = 조기종료 미발동, 상한에 막힘.
leaves63+iter1000 arm은 leaves와 iter를 동시에 바꿔 이 효과를 분리 못 했다.
또 용량을 '줄이는' 방향은 한 번도 시도 안 했는데, 늘리면 -143인 걸 보면 줄이는 쪽이 맞을 수 있다.
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from metrics import evaluate
from phase2_common import build_fold
from platoon import build_platoon_table, transform_platoon, K_PLATOON

SEED = 42
BASE = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
            early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)
INS = ["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]

def run(Xtr,ytr,Xva,yva,tag,**ov):
    p=dict(BASE,**ov); t=time.time()
    m=HistGradientBoostingClassifier(**p).fit(Xtr,ytr)
    b=evaluate(yva,m.predict_proba(Xva)[:,1])["bss"]
    print(f"  [{tag:28s}] iter={m.n_iter_:5d}/{p['max_iter']}  BSS={b:.6f}  score={max(0,b*100000):7.1f}  ({time.time()-t:.0f}s)",flush=True)
    return b

df=pd.read_csv("../data/train.csv",encoding="utf-8-sig")
df["row_num"]=df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g=float(df["control_success"].mean()); sr=sorted(df["season"].unique().tolist())
se=build_season_end_table(df); dins=transform_inseason(df,se,g,sr)
piv=_pivots_from_table(se,sr); idx=pd.MultiIndex.from_arrays([df["pitcher_id"],df["season"]-1])
pp=pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt=transform_platoon(df,build_platoon_table(df),pp,sr,k=K_PLATOON)
dinn=transform_inning(df,build_inning_table(df),build_inning_offset(df),pp,sr,k=570.0)
fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
def stack(bf,i): return pd.concat([bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
                                    dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True)],axis=1)
Xtr,Xva=stack(fold["X_train"],tr),stack(fold["X_valid"],va)
print(f"준비완료 {Xtr.shape[1]}피처\n"+"="*72,flush=True)
base=run(Xtr,ytr,Xva,yva,"baseline (leaves31 it500)")
res={}
res["it1500"]=run(Xtr,ytr,Xva,yva,"leaves31 it1500",max_iter=1500)
res["it1500_lr015"]=run(Xtr,ytr,Xva,yva,"leaves31 it1500 lr.015",max_iter=1500,learning_rate=0.015)
res["leaves15"]=run(Xtr,ytr,Xva,yva,"leaves15 it1000",max_leaf_nodes=15,max_iter=1000)
res["leaves15_d4"]=run(Xtr,ytr,Xva,yva,"leaves15 depth4 it1500",max_leaf_nodes=15,max_depth=4,max_iter=1500)
res["leaves8"]=run(Xtr,ytr,Xva,yva,"leaves8 it2000",max_leaf_nodes=8,max_iter=2000)
print("\n"+"="*72+"\nbaseline 대비 (실제예상 = 델타 x0.47)\n"+"="*72,flush=True)
for k,v in res.items():
    d=100000*(v-base); print(f"  {k:22s} {d:+7.1f}   실제예상={d*0.47:+6.1f}",flush=True)
