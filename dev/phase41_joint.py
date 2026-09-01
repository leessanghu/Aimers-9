"""작년 피처 7개의 '합동' 가치 계산 (개별 합산은 상관 때문에 과대계상).
최적 선형 개선량 = ||P_Z e||^2 / N  (P_Z = 잔차화된 Z 위로의 사영)"""
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
dly=transform_lastyear(df,build_lastyear_table(df),build_global_rates(df),sr,k=30.0)
fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
ti,ei=time_split_es(len(tr))
def stack(i,bf):
    X=pd.concat([bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
                 dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True),
                 dpt.loc[i].reset_index(drop=True)],axis=1).astype(np.float64)
    return pd.concat([X,add_crosses(X)],axis=1)
Xtr,Xva=stack(tr,fold["X_train"]),stack(va,fold["X_valid"])
h=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,
    l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,
    random_state=SEED).fit(Xtr,ytr)
cb=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=5.0,random_seed=SEED,
    verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
cb.fit(Xtr.iloc[ti],ytr[ti],eval_set=(Xtr.iloc[ei],ytr[ei]))
p=0.5*h.predict_proba(Xva)[:,1]+0.5*cb.predict_proba(Xva)[:,1]
e=yva-p; r=yva.mean(); bv=r*(1-r)
Xn=Xva.to_numpy(np.float64); Xd=np.column_stack([np.ones(len(Xn)),Xn,p,p**2])
XtX=Xd.T@Xd+1e-3*np.eye(Xd.shape[1])
Z=dly.loc[va].to_numpy(np.float64)
Zp=Z-Xd@np.linalg.solve(XtX,Xd.T@Z)                 # 전부 한꺼번에 잔차화
ZtZ=Zp.T@Zp+1e-8*np.eye(Zp.shape[1])
proj=Zp@np.linalg.solve(ZtZ,Zp.T@e)
joint=(proj@proj)/len(e)/bv*1e5
print(f"v14 구성 score={max(0,evaluate(yva,p)['bss']*1e5):.1f}")
print(f"\n작년 피처 7개 합동 예상이득 = {joint:+.1f}   (개별 합산 +21.5는 과대계상)")
# 하위집합도 확인
for cols in [["ly_reverse"],["ly_success","ly_reverse"],
             ["ly_success","ly_reverse","ly_minus_career"],
             ["ly_success","ly_reverse","ly_middle","ly_ball","ly_n"]]:
    Zs=dly.loc[va,cols].to_numpy(np.float64)
    Zps=Zs-Xd@np.linalg.solve(XtX,Xd.T@Zs)
    pr=Zps@np.linalg.solve(Zps.T@Zps+1e-8*np.eye(Zps.shape[1]),Zps.T@e)
    print(f"  {str(cols):58s} {(pr@pr)/len(e)/bv*1e5:+7.1f}")
print(f"\n총 {time.time()-t0:.0f}s")
