import sys, os
sys.stdout.reconfigure(encoding="utf-8"); sys.path.insert(0, os.getcwd())
import script as S, joblib, pandas as pd, numpy as np
def run(d):
    a=joblib.load("model/model_artifacts_v17a.pkl")
    t=S.load_test(os.path.join(d,"test.csv")); ids=t[S.ID_COL].tolist()
    Xb=S.build_features(t,a["stats"]).reset_index(drop=True)
    Xi=S.build_inseason_features(t,a["inseason_stats"]).reset_index(drop=True)
    pr=S.get_prior_pitcher_rate(t,a["inseason_stats"])
    Xp=S.build_platoon_features(t,a["platoon_stats"],pr).reset_index(drop=True)
    Xn=S.build_inning_features(t,a["inning_stats"],pr).reset_index(drop=True)
    Xt=S.build_pitchtype_features(t,a["pitchtype_stats"],pr).reset_index(drop=True)
    Xl=S.build_lastyear_features(t,a["lastyear_stats"]).reset_index(drop=True)
    cs=(t.balls_before*4+t.strikes_before).to_numpy()
    inn=np.clip(t.inning.to_numpy(np.int64),1,9)
    Xlc=S.build_label_cond_features(t,a["labels_c_stats"],cs,"lc").reset_index(drop=True)
    Xli=S.build_label_cond_features(t,a["labels_i_stats"],inn,"li").reset_index(drop=True)
    X=pd.concat([Xb,Xi,Xp,Xn,Xt],axis=1); X.index=t.index; X=X.astype(np.float64)
    Xc=S.add_crosses(X); Xl.index=t.index; Xlc.index=t.index; Xli.index=t.index
    X=pd.concat([X,Xc,Xl,Xlc,Xli],axis=1)[a["feature_order"]].astype(np.float64)
    p=a["w_hgb"]*a["hgb"].predict_proba(X)[:,1]+a["w_cat"]*a["cat"].predict_proba(X)[:,1]
    return dict(zip(ids,p))
o=run("../data"); s=run("_s/data"); b=run("_b/data")
m1=max(abs(o[k]-s[k]) for k in o); print(f"셔플     최대차이={m1:.2e}  {'PASS' if m1<1e-9 else 'FAIL'}")
m2=max(abs(o[k]-b[k]) for k in b); print(f"부분집합 최대차이={m2:.2e}  {'PASS' if m2<1e-9 else 'FAIL'}")
