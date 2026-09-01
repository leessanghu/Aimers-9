"""기존 67피처 제거 실험 — 16전 16패가 가리키는 유일한 미탐색 방향.

논리: 약한 구조적 피처는 난수보다 해롭다(우리가 직접 증명: 난수 -0.9 vs 실제피처 -6~-31).
      그런데 67개 중 58개는 in-season/platoon/inning 이전에 정해졌고 그 뒤 재검증이 없다.
      제거가 오히려 점수를 올릴 수 있고, 용량도 되돌려준다.
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

def run(Xtr,ytr,Xva,yva,tag):
    t=time.time(); m=HistGradientBoostingClassifier(**HGB).fit(Xtr,ytr)
    b=evaluate(yva,m.predict_proba(Xva)[:,1])["bss"]
    print(f"  [{tag:26s}] {Xtr.shape[1]:3d}피처  BSS={b:.6f}  score={max(0,b*100000):7.1f}  ({time.time()-t:.0f}s)",flush=True)
    return b

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
cols=list(Xtr.columns)
print(f"전체 {len(cols)}피처:\n{cols}\n"+"="*76,flush=True)

GROUPS={
 "team_te":[c for c in cols if c.endswith("_te")],
 "team_count":[c for c in cols if c.startswith(("pitcher_team_id_count","batter_team_id_count"))],
 "id_count":["pitcher_id_count","batter_id_count"],
 "prev_game(6)":[c for c in cols if "prev1_game" in c or "prev3_game" in c or "prev5_game" in c or c=="flag_prev_game_missing"],
 "pitchmix":[c for c in cols if "pitchmix" in c or "fastball" in c or "breaking" in c or "offspeed" in c],
 "batter_asof":[c for c in cols if "asof_batter" in c],
 "winexp_li":[c for c in cols if c in ("home_win_expectancy","away_win_expectancy","li")],
 "runs_score":[c for c in cols if c in ("run_top_before","run_bot_before","run_total_before","score_diff_home","score_diff_pitcher_team")],
 "runners":[c for c in cols if c in ("runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","cat_base_state")],
 "date":[c for c in cols if c in ("game_month","game_dayofweek","season")],
 "cat_misc":[c for c in cols if c in ("cat_top_bottom","cat_game_type")],
 "diff_pair":[c for c in cols if c in ("diff_success_rate","diff_middle_rate")],
}
base=run(Xtr,ytr,Xva,yva,"baseline(v7c)")
res={}
for nm,drop in GROUPS.items():
    drop=[c for c in drop if c in cols]
    if not drop: continue
    res[nm]=(run(Xtr.drop(columns=drop),ytr,Xva.drop(columns=drop),yva,f"-{nm}({len(drop)})"), len(drop))
print("\n"+"="*76+"\n제거 효과 (양수 = 빼는 게 이득)  실제예상 = 델타 x0.47\n"+"="*76,flush=True)
for nm,(v,k) in sorted(res.items(), key=lambda x:-x[1][0]):
    d=100000*(v-base); print(f"  -{nm:20s}({k:2d}개) {d:+7.1f}   실제예상={d*0.47:+6.1f}",flush=True)
