"""구종 피처 폴드 검증 — baseline = v12 실전 구성(963.796) 정확히 재현 + 구종 3개만 추가.
단일 변수 변경. 시드 노이즈(SD~11) 대응으로 3시드 평균 비교."""
import sys, time
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from catboost import CatBoostClassifier
from crosses import add_crosses
from inning_split import build_inning_offset, build_inning_table, transform_inning, K_INNING
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import build_platoon_table, transform_platoon, K_PLATOON
SEED=42
INS=["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]
df=pd.read_csv("../data/train.csv",encoding="utf-8-sig")
df["row_num"]=df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g=float(df["control_success"].mean()); sr=sorted(df["season"].unique().tolist())
se=build_season_end_table(df); dins=transform_inseason(df,se,g,sr)   # K=15 (v12 값)
piv=_pivots_from_table(se,sr); idx=pd.MultiIndex.from_arrays([df["pitcher_id"],df["season"]-1])
pp=pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt=transform_platoon(df,build_platoon_table(df),pp,sr,k=K_PLATOON)  # K=520 (v12 값)
dinn=transform_inning(df,build_inning_table(df),build_inning_offset(df),pp,sr,k=K_INNING)
mt=build_matched(df); tb=build_pitchtype_tables(mt,sr)
dpt=transform_pitchtype(df,tb,pp,g,sr)
print(f"매칭 {len(mt):,}  pt_dev SD={dpt.pt_dev.std():.5f}",flush=True)
fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
ti,ei=time_split_es(len(tr))
def stack(i,bf,extra=()):
    parts=[bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
           dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True)]
    parts+=[e.loc[i].reset_index(drop=True) for e in extra]
    X=pd.concat(parts,axis=1).astype(np.float64)
    return pd.concat([X,add_crosses(X)],axis=1)
sc=lambda p: max(0,evaluate(yva,p)["bss"]*1e5)
def run(extra,tag):
    xt,xv=stack(tr,fold["X_train"],extra),stack(va,fold["X_valid"],extra)
    h=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,
        l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,
        random_state=SEED).fit(xt,ytr)
    ph=h.predict_proba(xv)[:,1]
    ps=[]
    for s in (42,7,2024):
        c=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=5.0,random_seed=s,
            verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
        c.fit(xt.iloc[ti],ytr[ti],eval_set=(xt.iloc[ei],ytr[ei])); ps.append(c.predict_proba(xv)[:,1])
    pc=np.mean(ps,axis=0)
    print(f"  [{tag:12s}] {xt.shape[1]:3d}피처  hgb={sc(ph):7.1f}  cat3시드={sc(pc):7.1f}  "
          f"50:50={sc(0.5*ph+0.5*pc):7.1f}  개별={[round(sc(p),1) for p in ps]}",flush=True)
    return sc(0.5*ph+0.5*pc), sc(pc)
print("\n"+"="*84,flush=True)
b_bl,c_bl=run((),"baseline(v12)")
b_pt,c_pt=run((dpt,),"+구종3개")
print(f"\n블렌드 델타={b_pt-b_bl:+.1f}   CatBoost 델타={c_pt-c_bl:+.1f}",flush=True)
