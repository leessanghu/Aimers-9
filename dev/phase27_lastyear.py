"""작년 피처 + 합성 실력 추정치 검증 — 2024 폴드. baseline = v12 구성(963.796)."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from crosses import add_crosses
from inning_split import build_inning_offset, build_inning_table, transform_inning, K_INNING
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear, LASTYEAR_COLS
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from platoon import build_platoon_table, transform_platoon, K_PLATOON
SEED=42
INS=["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]
df=pd.read_csv("../data/train.csv",encoding="utf-8-sig")
df["row_num"]=df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g=float(df["control_success"].mean()); sr=sorted(df["season"].unique().tolist())
se=build_season_end_table(df); dins=transform_inseason(df,se,g,sr)
piv=_pivots_from_table(se,sr); idx=pd.MultiIndex.from_arrays([df["pitcher_id"],df["season"]-1])
pp=pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt=transform_platoon(df,build_platoon_table(df),pp,sr,k=K_PLATOON)
dinn=transform_inning(df,build_inning_table(df),build_inning_offset(df),pp,sr,k=K_INNING)
gr=build_global_rates(df); lyt=build_lastyear_table(df)
dly=transform_lastyear(df,lyt,gr,sr)
print("작년 피처 SD:", {c: round(float(dly[c].std()),4) for c in LASTYEAR_COLS}, flush=True)
m19=df.season==2019
print(f"2019행 ly_n 최대={dly.loc[m19,'ly_n'].max():.3f} (0이어야 정상)", flush=True)
fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
def stack(bf,i,extra=()):
    parts=[bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
           dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True)]
    parts+=[e.loc[i].reset_index(drop=True) for e in extra]
    X=pd.concat(parts,axis=1).astype(np.float64)
    return pd.concat([X,add_crosses(X)],axis=1)
def run(xt,xv,tag):
    t=time.time()
    h=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,
        l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,
        random_state=SEED).fit(xt,ytr)
    ph=h.predict_proba(xv)[:,1]
    ti,ei=time_split_es(len(xt))
    c=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=5.0,random_seed=SEED,
        verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
    c.fit(xt.iloc[ti],ytr[ti],eval_set=(xt.iloc[ei],ytr[ei]))
    pc=c.predict_proba(xv)[:,1]
    bh,bc=evaluate(yva,ph)["bss"],evaluate(yva,pc)["bss"]
    bb=evaluate(yva,0.5*ph+0.5*pc)["bss"]
    print(f"  [{tag:14s}] {xt.shape[1]:3d}피처  hgb={max(0,bh*1e5):7.1f}  cat={max(0,bc*1e5):7.1f}"
          f"  blend={max(0,bb*1e5):7.1f}  ({time.time()-t:.0f}s)",flush=True)
    return bb,bc
print("\n"+"="*74,flush=True)
b0,c0=run(stack(fold["X_train"],tr),stack(fold["X_valid"],va),"baseline(v12)")
b1,c1=run(stack(fold["X_train"],tr,[dly]),stack(fold["X_valid"],va,[dly]),"+작년7개")
print(f"\n블렌드 델타 = {1e5*(b1-b0):+.1f}   CatBoost 델타 = {1e5*(c1-c0):+.1f}",flush=True)
