"""최종 조합 확정 — 노이즈 기준(±5~8) 넘는 것만 채택하고 시드평균으로 노이즈 제거.

채택: K_inseason=60, platoon K↑, lastyear K=30, cat depth=6, l2=15, 3시드평균, 20:80
확인: platoon K를 더 올려도 되는지 (1200 -> 2500)
"""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import inseason as INS_MOD
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
gr=build_global_rates(df); dly=transform_lastyear(df,build_lastyear_table(df),gr,sr,k=30.0)
dinn=transform_inning(df,inn_tbl,inn_off,pp,sr,k=570.0)
fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
ti,ei=time_split_es(len(tr))
def build(kp):
    dplt=transform_platoon(df,plt_tbl,pp,sr,k=kp)
    out={}
    for nm,i,bf in [("tr",tr,fold["X_train"]),("va",va,fold["X_valid"])]:
        X=pd.concat([bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
                     dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True),
                     dly.loc[i].reset_index(drop=True)],axis=1).astype(np.float64)
        out[nm]=pd.concat([X,add_crosses(X)],axis=1)
    return out["tr"],out["va"]
def cat(xt,xv,seed,l2=15.0):
    c=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=l2,random_seed=seed,
        verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
    c.fit(xt.iloc[ti],ytr[ti],eval_set=(xt.iloc[ei],ytr[ei]))
    return c.predict_proba(xv)[:,1]
sc=lambda p: max(0,evaluate(yva,p)["bss"]*1e5)
for kp in [1200.0, 2500.0]:
    t=time.time(); xt,xv=build(kp)
    ps=[cat(xt,xv,s) for s in (42,7,2024)]
    print(f"platoon K={kp:6.0f}  개별={[round(sc(p),1) for p in ps]}  3시드평균={sc(np.mean(ps,axis=0)):8.1f}  ({time.time()-t:.0f}s)",flush=True)
    if kp==1200.0: best_ps, best_xt, best_xv = ps, xt, xv
print("\n[20:80 블렌드 확인]",flush=True)
h=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,
    l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,
    random_state=SEED).fit(best_xt,ytr)
ph=h.predict_proba(best_xv)[:,1]
pc=np.mean(best_ps,axis=0)
print(f"  hgb 단독 = {sc(ph):.1f}   cat 3시드 = {sc(pc):.1f}",flush=True)
for w in [0.0,0.1,0.15,0.2,0.3]:
    print(f"  hgb {w:.2f} : cat {1-w:.2f}  ->  {sc(w*ph+(1-w)*pc):8.1f}",flush=True)
