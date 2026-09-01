"""batter_asof 제거가 2022 폴드에서도 재현되는지 (2024에서만 +33.0 확인됨).
baseline = v7c 실전 구성 (실제 948.970점). HGB 단독, 다른 변경 없음."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from metrics import evaluate
from phase2_common import build_fold
from platoon import build_platoon_table, transform_platoon, K_PLATOON
SEED=42
HGB=dict(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,l2_regularization=5.0,
         early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,random_state=SEED)
INS=["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]
BA=["flag_asof_batter_n_zero","asof_batter_n","asof_batter_success_rate_smooth","asof_batter_middle_rate_smooth"]
df=pd.read_csv("../data/train.csv",encoding="utf-8-sig")
df["row_num"]=df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g=float(df["control_success"].mean()); sr=sorted(df["season"].unique().tolist())
se=build_season_end_table(df); dins=transform_inseason(df,se,g,sr)
piv=_pivots_from_table(se,sr); idx=pd.MultiIndex.from_arrays([df["pitcher_id"],df["season"]-1])
pp=pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt=transform_platoon(df,build_platoon_table(df),pp,sr,k=K_PLATOON)
dinn=transform_inning(df,build_inning_table(df),build_inning_offset(df),pp,sr,k=570.0)
for tmax,vs in [(2021,2022),(2023,2024)]:
    fold=build_fold(df,tmax,vs,extra_features=None,seed=SEED,include_team_te=True)
    ytr,yva=fold["y_train"],fold["y_valid"]
    tr,va=df[df.season<=tmax].index, df[df.season==vs].index
    def stack(bf,i): return pd.concat([bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
                                        dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True)],axis=1)
    Xtr,Xva=stack(fold["X_train"],tr),stack(fold["X_valid"],va)
    print(f"\n=== FOLD valid={vs} ===",flush=True)
    m=HistGradientBoostingClassifier(**HGB).fit(Xtr,ytr); b0=evaluate(yva,m.predict_proba(Xva)[:,1])["bss"]
    print(f"  baseline(v7c)   {Xtr.shape[1]}피처  score={max(0,b0*1e5):7.1f}",flush=True)
    d=[c for c in BA if c in Xtr.columns]
    m=HistGradientBoostingClassifier(**HGB).fit(Xtr.drop(columns=d),ytr)
    b1=evaluate(yva,m.predict_proba(Xva.drop(columns=d))[:,1])["bss"]
    print(f"  -batter_asof    {Xtr.shape[1]-len(d)}피처  score={max(0,b1*1e5):7.1f}   delta={1e5*(b1-b0):+7.1f}",flush=True)
