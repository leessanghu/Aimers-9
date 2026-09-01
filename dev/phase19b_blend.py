"""NN 예측과 HGB(v7c) 예측의 상관 + 블렌딩 이득 측정."""
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
INS=["inseason_success_smooth","inseason_ball_smooth","inseason_reverse_smooth","inseason_n","inseason_is_first_appearance"]
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
t=time.time()
hgb=HistGradientBoostingClassifier(max_depth=6,max_leaf_nodes=31,max_iter=500,learning_rate=0.03,
    l2_regularization=5.0,early_stopping=True,validation_fraction=0.1,n_iter_no_change=20,random_state=SEED).fit(Xtr,ytr)
p_hgb=hgb.predict_proba(Xva)[:,1]
np.save("phase19b_hgb_pred_2024.npy",p_hgb)
p_nn=np.load("phase19_nn_pred_2024.npy")
b_h=evaluate(yva,p_hgb)["bss"]; b_n=evaluate(yva,p_nn)["bss"]
print(f"HGB  BSS={b_h:.6f} ({max(0,b_h*1e5):.1f})   NN BSS={b_n:.6f} ({max(0,b_n*1e5):.1f})  ({time.time()-t:.0f}s)")
print(f"\n예측 상관 r = {np.corrcoef(p_hgb,p_nn)[0,1]:.4f}   <-- 0.95 이하면 앙상블 이득 기대")
print(f"  HGB pred SD={p_hgb.std():.4f}  NN pred SD={p_nn.std():.4f}")
print("\n블렌딩 (w = NN 비중):")
best=(0,b_h)
for w in [0,.05,.1,.15,.2,.25,.3,.4,.5]:
    p=(1-w)*p_hgb+w*p_nn; b=evaluate(yva,p)["bss"]; d=1e5*(b-b_h)
    if b>best[1]: best=(w,b)
    print(f"  w={w:.2f}  BSS={b:.6f}  score={max(0,b*1e5):7.1f}  delta={d:+7.1f}  실제예상={d*0.47:+6.1f}")
print(f"\n최적 w={best[0]:.2f}  delta={1e5*(best[1]-b_h):+.1f}")
