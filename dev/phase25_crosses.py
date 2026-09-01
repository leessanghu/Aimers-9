"""교차항 14개 검증 — 2024 폴드. baseline = v10 구성(66피처, 칼만+HGB/Cat).
로컬은 부호를 못 맞춘 전력이 있으므로 '크게 망가지는지'만 본다."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from crosses import add_crosses
from inning_split import build_inning_offset, build_inning_table, transform_inning, K_INNING
from inseason import build_season_end_table, transform_inseason, _pivots_from_table, K_SMOOTH
from kalman_ability import build_kalman_table, estimate_process_noise, transform_kalman
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from platoon import build_platoon_table, transform_platoon, K_PLATOON
SEED=42
df=pd.read_csv("../data/train.csv",encoding="utf-8-sig")
df["row_num"]=df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g=float(df["control_success"].mean()); sr=sorted(df["season"].unique().tolist())
se=build_season_end_table(df); dins=transform_inseason(df,se,g,sr)
piv=_pivots_from_table(se,sr); idx=pd.MultiIndex.from_arrays([df["pitcher_id"],df["season"]-1])
pp=pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
q=estimate_process_noise(df); th,P=build_kalman_table(df,sr,q,g)
ns=np.expm1(dins["inseason_n"].to_numpy(np.float64)); smv=dins["inseason_success_smooth"].to_numpy(np.float64)
raw=np.clip(np.where(ns>0,(smv*(ns+K_SMOOTH)-K_SMOOTH*pp)/np.maximum(ns,1e-9),np.nan),0,1)
dkal=transform_kalman(df,th,P,g,inseason_n=ns,inseason_rate=raw)
dplt=transform_platoon(df,build_platoon_table(df),pp,sr,k=K_PLATOON)
dinn=transform_inning(df,build_inning_table(df),build_inning_offset(df),pp,sr,k=K_INNING)
fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
def stack(bf,i): return pd.concat([bf.reset_index(drop=True),dkal.loc[i].reset_index(drop=True),
                                    dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True)],axis=1).astype(np.float64)
Xtr,Xva=stack(fold["X_train"],tr),stack(fold["X_valid"],va)
Ctr,Cva=add_crosses(Xtr),add_crosses(Xva)
print(f"baseline {Xtr.shape[1]}피처 / +교차 {Xtr.shape[1]+Ctr.shape[1]}피처",flush=True)
print("교차항 SD:", {c: round(float(Cva[c].std()),4) for c in Ctr.columns}, flush=True)
def run(xt,xv,tag):
    t=time.time(); h=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,
        learning_rate=0.03,l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,
        n_iter_no_change=20,random_state=SEED).fit(xt,ytr)
    ph=h.predict_proba(xv)[:,1]
    ti,ei=time_split_es(len(xt))
    c=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=5.0,random_seed=SEED,
        verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
    c.fit(xt.iloc[ti],ytr[ti],eval_set=(xt.iloc[ei],ytr[ei]))
    pc=c.predict_proba(xv)[:,1]
    b=evaluate(yva,0.5*ph+0.5*pc)["bss"]
    print(f"  [{tag:16s}] {xt.shape[1]:3d}피처  hgb={evaluate(yva,ph)['bss']:.6f}  cat={evaluate(yva,pc)['bss']:.6f}"
          f"  blend={b:.6f} score={max(0,b*1e5):7.1f}  ({time.time()-t:.0f}s)",flush=True)
    return b
b0=run(Xtr,Xva,"baseline(v10)")
b1=run(pd.concat([Xtr,Ctr],axis=1),pd.concat([Xva,Cva],axis=1),"+교차14")
print(f"\n델타 = {1e5*(b1-b0):+.1f}",flush=True)
