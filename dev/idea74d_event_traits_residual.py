"""과거 PA-event 성향을 새 batter/pitcher trait로 만든 v66 잔차 실험."""

import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def build_event(df):
    d=df.sort_values("row_num").reset_index(); b=d.balls_before.to_numpy(); s=d.strikes_before.to_numpy()
    same=np.r_[(d.pitcher_id.to_numpy()[1:]==d.pitcher_id.to_numpy()[:-1]) &
               (d.batter_id.to_numpy()[1:]==d.batter_id.to_numpy()[:-1]) &
               (d.inning.to_numpy()[1:]==d.inning.to_numpy()[:-1]) &
               (d.top_bottom.to_numpy()[1:]==d.top_bottom.to_numpy()[:-1]),False]
    bn=np.r_[b[1:],-99]; sn=np.r_[s[1:],-99]; e=np.full(len(d),3,dtype=np.int8)
    e[same&(bn==b+1)&(sn==s)]=0; e[same&(sn==s+1)&(bn==b)]=1; e[same&(bn==b)&(sn==s)&(s==2)]=2
    out=np.empty(len(d),dtype=np.int8); out[d["index"].to_numpy()]=e; return out


def v66_valid():
    avg=lambda ps:np.mean([np.load(q) for q in ps],axis=0)
    base=avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6","d8","sub")])
    hur=np.mean([(1-np.load(f"phase90_cache/A_core_{n}.npy"))*np.load(f"phase90_cache/A_snc_{n}.npy") for n in ("d6","d8")],axis=0)
    return (.1824*base+.2432*hur+.0608*avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42,7)])
            +.1216*avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42,7)])
            +.1520*avg([f"idea46_cache/A_midother_s{s}.npy" for s in (42,7)])
            +.08*avg([f"idea54_cache/A_cond_ball_s{s}.npy" for s in (42,7)])
            +.08*avg([f"idea54_cache/A_count_resid_s{s}.npy" for s in (42,7)])
            +.08*avg([f"idea54_cache/A_future50_multi_s{s}.npy" for s in (42,7)]))


def score(y,p): return 1e5*(1-np.mean((p-y)**2)/(y.mean()*(1-y.mean())))


def entity_probs(tr,va,entity,prior,k):
    tab=tr.groupby([entity,"count","event"]).size().unstack("event",fill_value=0).reindex(columns=range(4),fill_value=0)
    ix=pd.MultiIndex.from_arrays([va[entity],va["count"]]); C=tab.reindex(ix).fillna(0).to_numpy(float); n=C.sum(1,keepdims=True)
    return (C+k*prior)/(n+k)


def crossfit_linear(y,p,Z,order,ridge=1e-3):
    pred=p.copy(); half=order<np.median(order)
    for fit,val in ((half,~half),(~half,half)):
        A=Z[fit]; mu=A.mean(0); sd=A.std(0); sd[sd<1e-12]=1
        Af=(A-mu)/sd; Av=(Z[val]-mu)/sd; target=y[fit]-p[fit]
        X=np.column_stack([np.ones(len(Af)),Af]); Xv=np.column_stack([np.ones(len(Av)),Av])
        coef=np.linalg.solve(X.T@X+ridge*np.eye(X.shape[1]),X.T@target)
        pred[val]+=Xv@coef
    return pred


use=["row_id","season","inning","top_bottom","balls_before","strikes_before","pitcher_id","batter_id","control_success"]
df=pd.read_csv("../data/train.csv",usecols=use,encoding="utf-8-sig"); df["row_num"]=df.row_id.str[6:].astype(int)
df["count"]=df.balls_before*3+df.strikes_before; df["event"]=build_event(df)
tr=df[df.season<=2023]; va=df[df.season==2024]; y=va.control_success.to_numpy(float); p=v66_valid(); base=score(y,p)
z=tr.groupby(["count","event"]).size().unstack("event",fill_value=0).reindex(index=range(12),columns=range(4),fill_value=0).to_numpy(float)
prior=(z/z.sum(1,keepdims=True))[va["count"].to_numpy()]; order=va.row_num.to_numpy()
rows=[]
q0=crossfit_linear(y,p,np.empty((len(va),0)),order)
intercept_delta=score(y,q0)-base
print(f"intercept-only crossfit delta={intercept_delta:+.6f}")
for k in (20.,50.,100.,200.,500.,1000.,3000.):
    pp=entity_probs(tr,va,"pitcher_id",prior,k)-prior; pb=entity_probs(tr,va,"batter_id",prior,k)-prior
    blocks={"pitcher4":pp,"batter4":pb,"both8":np.column_stack([pp,pb]),"batter_paend":pb[:,[3]],"batter_foul_paend":pb[:,[2,3]]}
    for name,Z in blocks.items():
        q=crossfit_linear(y,p,Z,order); delta=score(y,q)-base
        rows.append((k,name,delta,delta-intercept_delta,np.corrcoef(q-p,y-p)[0,1]))
out=pd.DataFrame(rows,columns=["K","block","delta","net_vs_intercept","corr_correction_resid"]).sort_values("net_vs_intercept",ascending=False)
print(f"v66={base:.3f}")
print(out.to_string(index=False))
out.to_csv("idea74d_event_traits_results.csv",index=False)
