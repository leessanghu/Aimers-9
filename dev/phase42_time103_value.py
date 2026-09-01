"""time103 12개 잔차 가치 재평가. baseline = v15 구성(976.099 실증).
검증된 지표(phase39): 구종 +5.6->실제+6.7 / workload,form +0.4->실제 가치없음."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from time103 import transform_time103, TIME103_COLS
SEED=42; t0=time.time()
INS=["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]
df=pd.read_csv("../data/train.csv",encoding="utf-8-sig")
df["row_num"]=df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g=float(df["control_success"].mean()); sr=sorted(df["season"].unique().tolist())
se=build_season_end_table(df); dins=transform_inseason(df,se,g,sr)
piv=_pivots_from_table(se,sr); idx=pd.MultiIndex.from_arrays([df["pitcher_id"],df["season"]-1])
pp=pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt=transform_platoon(df,build_platoon_table(df),pp,sr,k=K_PLATOON)
dinn=transform_inning(df,build_inning_table(df),build_inning_offset(df),pp,sr,k=K_INNING)
dpt=transform_pitchtype(df,build_pitchtype_tables(build_matched(df),sr),pp,g,sr)
gr=build_global_rates(df); ly_tbl=build_lastyear_table(df)
dly=transform_lastyear(df,ly_tbl,gr,sr,k=30.0)
dtime=transform_time103(df,ly_tbl,gr,sr)
print(f"v15 피처 준비 ({time.time()-t0:.0f}s)",flush=True)
fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
ti,ei=time_split_es(len(tr))
def stack(i,bf):
    X=pd.concat([bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
                 dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True),
                 dpt.loc[i].reset_index(drop=True)],axis=1).astype(np.float64)
    return pd.concat([X,add_crosses(X),dly.loc[i].reset_index(drop=True)],axis=1)
Xtr,Xva=stack(tr,fold["X_train"]),stack(va,fold["X_valid"])
h=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,
    l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,
    random_state=SEED).fit(Xtr,ytr)
cb=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=5.0,random_seed=SEED,
    verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
cb.fit(Xtr.iloc[ti],ytr[ti],eval_set=(Xtr.iloc[ei],ytr[ei]))
p=0.5*h.predict_proba(Xva)[:,1]+0.5*cb.predict_proba(Xva)[:,1]
print(f"v15 구성 score={max(0,evaluate(yva,p)['bss']*1e5):.1f}",flush=True)
e=yva-p; r=yva.mean(); bv=r*(1-r)
Xn=Xva.to_numpy(np.float64); Xd=np.column_stack([np.ones(len(Xn)),Xn,p,p**2])
XtX=Xd.T@Xd+1e-3*np.eye(Xd.shape[1])
def value_each(zdf,tag):
    tot=0.0
    for col in zdf.columns:
        z=zdf[col].to_numpy(np.float64)
        if z.std()<1e-12: continue
        zp=z-Xd@np.linalg.solve(XtX,Xd.T@z); v=zp.var()
        if v<1e-14: print(f"    {col:24s} (전부 설명됨)"); continue
        gain=(np.cov(e,zp)[0,1]**2/v)/bv*1e5; tot+=gain
        print(f"    {col:24s} corr={np.corrcoef(e,zp)[0,1]:+.5f}  예상이득={gain:+7.2f}",flush=True)
    print(f"  [{tag}] 개별합산 = {tot:+.1f}\n",flush=True)
def value_joint(zdf,tag):
    Z=zdf.to_numpy(np.float64)
    Zp=Z-Xd@np.linalg.solve(XtX,Xd.T@Z)
    proj=Zp@np.linalg.solve(Zp.T@Zp+1e-8*np.eye(Zp.shape[1]),Zp.T@e)
    joint=(proj@proj)/len(e)/bv*1e5
    print(f"  [{tag}] 합동 = {joint:+.1f}\n",flush=True)
print("="*74+"\ntime103 12개 개별 잔차가치\n"+"="*74,flush=True)
value_each(dtime.loc[va],"time103 전체")
print("="*74+"\n축별 합동가치 (success/reverse/ball/middle 각 3개씩)\n"+"="*74,flush=True)
for key in ["success","reverse","ball","middle"]:
    cols=[c for c in TIME103_COLS if key in c]
    value_joint(dtime.loc[va,cols], f"{key} 축 3개")
print("="*74+"\n유형별 합동가치 (rel / cur_minus_ly / ly_minus_old)\n"+"="*74,flush=True)
for pat,nm in [("_rel","ly_*_rel (리그보정)"),("cur_minus_ly","cur_minus_ly_* (올해vs작년 트렌드)"),
               ("ly_minus_old","ly_minus_old_* (작년vs재작년 트렌드)")]:
    cols=[c for c in TIME103_COLS if pat in c]
    value_joint(dtime.loc[va,cols], nm)
value_joint(dtime.loc[va,TIME103_COLS], "12개 전체 합동")
print(f"총 {time.time()-t0:.0f}s",flush=True)
