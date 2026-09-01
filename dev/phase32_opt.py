"""나머지 K들 + CatBoost 자체 최적화. in-season K가 10배 틀렸으니 나머지도 의심.

고정: K_inseason=60, 작년피처 on, 교차항 on, 평가는 CatBoost 단독(블렌드의 80%)
스윕: platoon K(현재520) / inning K(현재570) / lastyear K(현재30) / cat depth / 시드평균
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import inseason as INS_MOD, lastyear as LY_MOD
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from crosses import add_crosses
from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from platoon import build_platoon_table, transform_platoon
SEED=42
INS=["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]
df=pd.read_csv("../data/train.csv",encoding="utf-8-sig")
df["row_num"]=df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g=float(df["control_success"].mean()); sr=sorted(df["season"].unique().tolist())
se=build_season_end_table(df)
piv=_pivots_from_table(se,sr); idx=pd.MultiIndex.from_arrays([df["pitcher_id"],df["season"]-1])
pp=pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
INS_MOD.K_SMOOTH=60.0
dins=INS_MOD.transform_inseason(df,se,g,sr)
plt_tbl=build_platoon_table(df); inn_tbl=build_inning_table(df); inn_off=build_inning_offset(df)
gr=build_global_rates(df); ly_tbl=build_lastyear_table(df)
fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
ti,ei=None,None
def build(kp,ki,kl):
    dplt=transform_platoon(df,plt_tbl,pp,sr,k=kp)
    dinn=transform_inning(df,inn_tbl,inn_off,pp,sr,k=ki)
    dly=transform_lastyear(df,ly_tbl,gr,sr,k=kl)
    out={}
    for nm,i,bf in [("tr",tr,fold["X_train"]),("va",va,fold["X_valid"])]:
        X=pd.concat([bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
                     dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True),
                     dly.loc[i].reset_index(drop=True)],axis=1).astype(np.float64)
        out[nm]=pd.concat([X,add_crosses(X)],axis=1)
    return out["tr"],out["va"]
def cat(xt,xv,seed=SEED,depth=6,l2=5.0):
    global ti,ei
    if ti is None: ti,ei=time_split_es(len(xt))
    c=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=depth,l2_leaf_reg=l2,random_seed=seed,
        verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
    c.fit(xt.iloc[ti],ytr[ti],eval_set=(xt.iloc[ei],ytr[ei]))
    return c.predict_proba(xv)[:,1]
sc=lambda p: max(0,evaluate(yva,p)["bss"]*1e5)
BP,BI,BL=520.0,570.0,30.0
xt,xv=build(BP,BI,BL); base=cat(xt,xv)
print(f"기준 (plat520/inn570/ly30, depth6)  cat={sc(base):.1f}\n",flush=True)
print("[platoon K]",flush=True)
for k in [150.0,1200.0]:
    a,b=build(k,BI,BL); print(f"  K={k:6.0f}  {sc(cat(a,b)):8.1f}",flush=True)
print("[inning K]",flush=True)
for k in [200.0,1500.0]:
    a,b=build(BP,k,BL); print(f"  K={k:6.0f}  {sc(cat(a,b)):8.1f}",flush=True)
print("[lastyear K]",flush=True)
for k in [10.0,100.0]:
    a,b=build(BP,BI,k); print(f"  K={k:6.0f}  {sc(cat(a,b)):8.1f}",flush=True)
print("[cat depth]",flush=True)
for d in [5,7]:
    print(f"  depth={d}  {sc(cat(xt,xv,depth=d)):8.1f}",flush=True)
print("[cat l2]",flush=True)
for l in [1.0,15.0]:
    print(f"  l2={l:4.1f}   {sc(cat(xt,xv,l2=l)):8.1f}",flush=True)
print("[시드 평균]",flush=True)
ps=[base]+[cat(xt,xv,seed=s) for s in (7,2024)]
print(f"  단일={sc(base):.1f}  3시드평균={sc(np.mean(ps,axis=0)):.1f}",flush=True)
