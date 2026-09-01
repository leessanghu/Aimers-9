"""in-season K 스윕 + 모델 비중 — 2024 폴드.
phase30: 이론최적 K=87, 경험최적 K=150. 현재 15 (10배 과소축소).
baseline = v12 구성 + 작년피처(phase27에서 cat +15.4 확인)."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
import inseason as INS_MOD
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from crosses import add_crosses
from inning_split import build_inning_offset, build_inning_table, transform_inning, K_INNING
from inseason import build_season_end_table, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from platoon import build_platoon_table, transform_platoon, K_PLATOON
SEED=42
INS=["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]
df=pd.read_csv("../data/train.csv",encoding="utf-8-sig")
df["row_num"]=df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g=float(df["control_success"].mean()); sr=sorted(df["season"].unique().tolist())
se=build_season_end_table(df)
piv=_pivots_from_table(se,sr); idx=pd.MultiIndex.from_arrays([df["pitcher_id"],df["season"]-1])
pp=pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt=transform_platoon(df,build_platoon_table(df),pp,sr,k=K_PLATOON)
dinn=transform_inning(df,build_inning_table(df),build_inning_offset(df),pp,sr,k=K_INNING)
gr=build_global_rates(df); dly=transform_lastyear(df,build_lastyear_table(df),gr,sr)
fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
def build(dins,i,bf):
    X=pd.concat([bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
                 dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True),
                 dly.loc[i].reset_index(drop=True)],axis=1).astype(np.float64)
    return pd.concat([X,add_crosses(X)],axis=1)
print(f"{'K':>5s} {'hgb':>8s} {'cat':>8s} {'50:50':>8s} {'20:80':>8s} {'cat단독':>8s}",flush=True)
for K in [15,60,150,300]:
    INS_MOD.K_SMOOTH=float(K)
    dins=INS_MOD.transform_inseason(df,se,g,sr)
    xt,xv=build(dins,tr,fold["X_train"]),build(dins,va,fold["X_valid"])
    h=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,
        l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,
        random_state=SEED).fit(xt,ytr)
    ph=h.predict_proba(xv)[:,1]
    ti,ei=time_split_es(len(xt))
    c=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=5.0,random_seed=SEED,
        verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
    c.fit(xt.iloc[ti],ytr[ti],eval_set=(xt.iloc[ei],ytr[ei]))
    pc=c.predict_proba(xv)[:,1]
    f=lambda p: max(0,evaluate(yva,p)["bss"]*1e5)
    print(f"{K:5d} {f(ph):8.1f} {f(pc):8.1f} {f(0.5*ph+0.5*pc):8.1f} {f(0.2*ph+0.8*pc):8.1f} {f(pc):8.1f}",flush=True)
