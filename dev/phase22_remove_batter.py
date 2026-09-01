"""타자 정보 통합 제거 — ablation 양수 3개를 합쳤을 때 시너지가 있는지.

ablation 결과 양수는 3개뿐: -batter_asof +33.0 / -diff_pair +4.1 / -team_te +0.4
이 중 diff_pair(diff_success_rate, diff_middle_rate)는 타자 스무딩값으로 만든 파생이라
batter_asof와 같은 축이다 -> 함께 빼면 '타자 정보 완전 제거'가 된다.

최종 후보 피처셋의 HGB 예측을 저장해서 NN 블렌딩까지 이어붙인다.
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
SEED=42
HGB=dict(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,l2_regularization=5.0,
         early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,random_state=SEED)
INS=["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]
BATTER_ASOF=["flag_asof_batter_n_zero","asof_batter_n","asof_batter_success_rate_smooth","asof_batter_middle_rate_smooth"]
DIFF_PAIR=["diff_success_rate","diff_middle_rate"]
TEAM_TE=["pitcher_team_id_te","batter_team_id_te"]

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
def run(drop,tag,save=None):
    d=[c for c in drop if c in Xtr.columns]
    xt,xv=Xtr.drop(columns=d),Xva.drop(columns=d)
    t=time.time(); m=HistGradientBoostingClassifier(**HGB).fit(xt,ytr)
    p=m.predict_proba(xv)[:,1]; b=evaluate(yva,p)["bss"]
    if save: np.save(save,p)
    print(f"  [{tag:34s}] {xt.shape[1]:3d}피처  BSS={b:.6f}  score={max(0,b*1e5):7.1f}  ({time.time()-t:.0f}s)",flush=True)
    return b
print("="*78+"\n타자 정보 제거 조합 (baseline=v7c 67피처, 실제 948.970)\n"+"="*78,flush=True)
base=run([], "baseline(v7c)", save="p22_base.npy")
res={}
res["-batter_asof"]=run(BATTER_ASOF,"-batter_asof")
res["-batter_asof -diff_pair"]=run(BATTER_ASOF+DIFF_PAIR,"-batter_asof -diff_pair",save="p22_nb_nd.npy")
res["-batter_asof -diff -team_te"]=run(BATTER_ASOF+DIFF_PAIR+TEAM_TE,"-batter_asof -diff -team_te",save="p22_nb_nd_nt.npy")
res["-diff_pair only"]=run(DIFF_PAIR,"-diff_pair only")
print("\n"+"="*78+"\nbaseline 대비 (실제예상 = 델타 x0.47)\n"+"="*78,flush=True)
for k,v in sorted(res.items(),key=lambda x:-x[1]):
    d=1e5*(v-base); print(f"  {k:32s} {d:+7.1f}   실제예상={d*0.47:+6.1f}",flush=True)
