"""구종 완전복원 vs 기존 Trackman매칭 — 잔차가치 비교. baseline = v15 구성(976.099 실증)."""
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
from pitchtype_exact import (build_control_table, build_mix_table, global_type_rates,
                             recover_pitchtype_labels, transform_pitchtype_exact)
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
gr=build_global_rates(df); ly_tbl=build_lastyear_table(df)
dly=transform_lastyear(df,ly_tbl,gr,sr,k=30.0)

print("기존 Trackman 매칭 구종...",flush=True)
dpt_old = transform_pitchtype(df,build_pitchtype_tables(build_matched(df),sr),pp,g,sr)

print("신규 완전복원 구종...",flush=True)
pt_labels = recover_pitchtype_labels(df)
valid = pt_labels["pt_fastball"].notna().mean()
print(f"  라벨 유효율={valid*100:.2f}%",flush=True)
cs = (df["balls_before"]*4+df["strikes_before"]).to_numpy()
ctrl_tbl = build_control_table(df, pt_labels)
mix_tbl = build_mix_table(df, pt_labels, cs)
gtr = global_type_rates(df, pt_labels)
dpt_new = transform_pitchtype_exact(df, ctrl_tbl, mix_tbl, gtr, pp, cs, sr)
print(f"  pt_dev(신규) SD={dpt_new.pt_dev.std():.5f}  (기존 SD={dpt_old.pt_dev.std():.5f})  ({time.time()-t0:.0f}s)",flush=True)

fold=build_fold(df,2023,2024,extra_features=None,seed=SEED,include_team_te=True)
ytr,yva=fold["y_train"],fold["y_valid"]
tr,va=df[df.season<=2023].index, df[df.season==2024].index
ti,ei=time_split_es(len(tr))
def stack(i,bf,dptx):
    X=pd.concat([bf.reset_index(drop=True),dins.loc[i,INS].reset_index(drop=True),
                 dplt.loc[i].reset_index(drop=True),dinn.loc[i].reset_index(drop=True),
                 dptx.loc[i].reset_index(drop=True)],axis=1).astype(np.float64)
    return pd.concat([X,add_crosses(X),dly.loc[i].reset_index(drop=True)],axis=1)
Xtr,Xva=stack(tr,fold["X_train"],dpt_old),stack(va,fold["X_valid"],dpt_old)
h=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,
    l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,
    random_state=SEED).fit(Xtr,ytr)
cb=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=5.0,random_seed=SEED,
    verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
cb.fit(Xtr.iloc[ti],ytr[ti],eval_set=(Xtr.iloc[ei],ytr[ei]))
p=0.5*h.predict_proba(Xva)[:,1]+0.5*cb.predict_proba(Xva)[:,1]
print(f"\nv15(기존구종) 구성 score={max(0,evaluate(yva,p)['bss']*1e5):.1f}  ({time.time()-t0:.0f}s)",flush=True)
e=yva-p; r=yva.mean(); bv=r*(1-r)
Xn=Xva.to_numpy(np.float64); Xd=np.column_stack([np.ones(len(Xn)),Xn,p,p**2])
XtX=Xd.T@Xd+1e-3*np.eye(Xd.shape[1])
def screen(zdf,tag):
    zdf=zdf.loc[va]
    keep=[c for c in zdf.columns if zdf[c].std()>1e-12]
    Z=zdf[keep].to_numpy(np.float64)
    Zp=Z-Xd@np.linalg.solve(XtX,Xd.T@Z)
    proj=Zp@np.linalg.solve(Zp.T@Zp+1e-8*np.eye(Zp.shape[1]),Zp.T@e)
    joint=(proj@proj)/len(e)/bv*1e5
    print(f"  [{tag}] 합동 = {joint:+.1f}",flush=True)
    for c in keep:
        z=zdf[c].to_numpy(np.float64); zp=z-Xd@np.linalg.solve(XtX,Xd.T@z); v=zp.var()
        if v<1e-14: continue
        print(f"    {c:12s} corr={np.corrcoef(e,zp)[0,1]:+.5f}  개별={(np.cov(e,zp)[0,1]**2/v)/bv*1e5:+6.2f}",flush=True)
print("\n=== 잔차가치: v15에 이미 들어있는 (기존 Trackman) 구종의 잔여가치 ===",flush=True)
screen(dpt_old,"기존(Trackman)")
print("\n=== 잔차가치: 신규(완전복원) 구종이 v15 잔차에 남기는 가치 (기존 대체 시 순이득 추정) ===",flush=True)
screen(dpt_new,"신규(완전복원)")
print(f"\n총 {time.time()-t0:.0f}s",flush=True)
