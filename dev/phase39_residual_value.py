"""잔차 기반 피처 가치 측정기 + 아는 답으로 검증.

Codex 제안(§2): 피처 가치는 진폭/재현상관이 아니라 '기존 모델 잔차와의 공분산'으로 결정.
  e = y - p,   z_perp = z - E[z | 기존 정보],   dBS = Cov(e,z_perp)^2 / Var(z_perp)
  score_gain = dBS / base_var * 1e5

먼저 이 지표 자체를 신뢰할 수 있는지 검증한다. 아는 실측 답:
  구종(pt_dev 등 3개)  실제 +6.7   <- v12 잔차로 재면 이게 나와야 함
  workload/form        실제 -4.2
  batter_asof(4개)     제거시 -26.6 (즉 가치 있음)
지표가 이 순서를 재현 못 하면 지표도 폐기.
"""
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
from pitchcount_recover import build_workload_features
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import build_platoon_table, transform_platoon, K_PLATOON

SEED=42
INS=["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]
t0=time.time()
df=pd.read_csv("../data/train.csv",encoding="utf-8-sig")
df["row_num"]=df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g=float(df["control_success"].mean()); sr=sorted(df["season"].unique().tolist())
se=build_season_end_table(df); dins=transform_inseason(df,se,g,sr)
piv=_pivots_from_table(se,sr); idx=pd.MultiIndex.from_arrays([df["pitcher_id"],df["season"]-1])
pp=pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt=transform_platoon(df,build_platoon_table(df),pp,sr,k=K_PLATOON)
dinn=transform_inning(df,build_inning_table(df),build_inning_offset(df),pp,sr,k=K_INNING)
mt=build_matched(df); dpt=transform_pitchtype(df,build_pitchtype_tables(mt,sr),pp,g,sr)
dwl=build_workload_features(df)
dform=pd.DataFrame({
    "form_prev1": np.nan_to_num(df.asof_pitcher_prev1_game_success_rate.to_numpy(np.float64)-dins.inseason_success_smooth.to_numpy(),nan=0.0),
    "form_prev5": np.nan_to_num(df.asof_pitcher_prev5_game_success_rate.to_numpy(np.float64)-dins.inseason_success_smooth.to_numpy(),nan=0.0),
}, index=df.index)
print(f"피처 준비 ({time.time()-t0:.0f}s)",flush=True)

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

# v12 구성(구종 없음)으로 잔차 생성
Xtr,Xva=stack(tr,fold["X_train"]),stack(va,fold["X_valid"])
h=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,
    l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,
    random_state=SEED).fit(Xtr,ytr)
c=CatBoostClassifier(iterations=3000,learning_rate=0.03,depth=6,l2_leaf_reg=5.0,random_seed=SEED,
    verbose=0,early_stopping_rounds=50,min_data_in_leaf=200,loss_function="Logloss")
c.fit(Xtr.iloc[ti],ytr[ti],eval_set=(Xtr.iloc[ei],ytr[ei]))
p=0.5*h.predict_proba(Xva)[:,1]+0.5*c.predict_proba(Xva)[:,1]
np.save("phase39_p_v12.npy",p)
base=evaluate(yva,p)["bss"]
print(f"v12 구성 잔차 생성 완료  score={max(0,base*1e5):.1f}  ({time.time()-t0:.0f}s)\n",flush=True)

e = yva - p
r = yva.mean(); base_var = r*(1-r)
Xn = Xva.to_numpy(np.float64)
Xd = np.column_stack([np.ones(len(Xn)), Xn, p, p**2])           # 기존 정보 = 전체 피처 + 예측
XtX = Xd.T@Xd + 1e-3*np.eye(Xd.shape[1])
def value(zdf, tag):
    tot=0.0; lines=[]
    for col in zdf.columns:
        z = zdf[col].to_numpy(np.float64)
        if z.std()<1e-12: continue
        beta = np.linalg.solve(XtX, Xd.T@z)
        zp = z - Xd@beta                                         # z_perp
        v = zp.var()
        if v<1e-14: lines.append(f"    {col:26s} (전부 설명됨)"); continue
        cov = np.cov(e, zp)[0,1]
        d = cov**2/v
        gain = d/base_var*1e5
        tot += gain
        lines.append(f"    {col:26s} z_perp SD={np.sqrt(v):.5f}  corr(e,z_perp)={np.corrcoef(e,zp)[0,1]:+.5f}  예상이득={gain:+7.2f}")
    print(f"  [{tag}]  합계 예상이득 = {tot:+.1f}",flush=True)
    for l in lines: print(l,flush=True)
    return tot

print("=== 지표 검증: 아는 실측 답과 비교 ===",flush=True)
value(dpt.loc[va], "구종 3개  (실제 +6.7)")
value(pd.concat([dwl.loc[va],dform.loc[va]],axis=1), "workload/form  (실제 -4.2)")
ba=[c_ for c_ in ["flag_asof_batter_n_zero","asof_batter_n","asof_batter_success_rate_smooth","asof_batter_middle_rate_smooth"] if c_ in Xva.columns]
print(f"\n  (참고) batter_asof는 이미 모델 안에 있어 z_perp≈0이 정상. 제거 실측은 -26.6",flush=True)
print(f"\n총 {time.time()-t0:.0f}s",flush=True)
