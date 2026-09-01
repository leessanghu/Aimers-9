"""pitcher×count PA-event trait의 A/B/C 3폴드 잔차 안정성."""
import sys
import numpy as np
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")

def event_label(df):
    d=df.sort_values("row_num").reset_index(); b=d.balls_before.to_numpy(); s=d.strikes_before.to_numpy()
    same=np.r_[(d.pitcher_id.to_numpy()[1:]==d.pitcher_id.to_numpy()[:-1])&(d.batter_id.to_numpy()[1:]==d.batter_id.to_numpy()[:-1])&(d.inning.to_numpy()[1:]==d.inning.to_numpy()[:-1])&(d.top_bottom.to_numpy()[1:]==d.top_bottom.to_numpy()[:-1]),False]
    bn=np.r_[b[1:],-9]; sn=np.r_[s[1:],-9]; e=np.full(len(d),3,dtype=np.int8)
    e[same&(bn==b+1)&(sn==s)]=0; e[same&(sn==s+1)&(bn==b)]=1; e[same&(bn==b)&(sn==s)&(s==2)]=2
    z=np.empty(len(d),dtype=np.int8); z[d["index"].to_numpy()]=e; return z

def pred(tag):
    avg=lambda ps:np.mean([np.load(q) for q in ps],axis=0)
    base=avg([f"phase90_cache/{tag}_base_{n}.npy" for n in ("d6","d8","sub")])
    hur=np.mean([(1-np.load(f"phase90_cache/{tag}_core_{n}.npy"))*np.load(f"phase90_cache/{tag}_snc_{n}.npy") for n in ("d6","d8")],axis=0)
    return (.1824*base+.2432*hur+.0608*avg([f"idea13_cache/{tag}_multires_s{s}.npy" for s in (42,7)])+.1216*avg([f"idea13_cache/{tag}_ordinal_s{s}.npy" for s in (42,7)])+.1520*avg([f"idea46_cache/{tag}_midother_s{s}.npy" for s in (42,7)])+.08*avg([f"idea54_cache/{tag}_cond_ball_s{s}.npy" for s in (42,7)])+.08*avg([f"idea54_cache/{tag}_count_resid_s{s}.npy" for s in (42,7)])+.08*avg([f"idea54_cache/{tag}_future50_multi_s{s}.npy" for s in (42,7)]))

def score(y,p): return 1e5*(1-np.mean((p-y)**2)/(y.mean()*(1-y.mean())))

def cf(y,p,Z,order):
    out=p.copy(); h=order<np.median(order)
    for fit,val in ((h,~h),(~h,h)):
        A=Z[fit]; mu=A.mean(0); sd=A.std(0); sd[sd<1e-12]=1; X=(A-mu)/sd; Xv=(Z[val]-mu)/sd
        X=np.column_stack([np.ones(len(X)),X]); Xv=np.column_stack([np.ones(len(Xv)),Xv]); t=y[fit]-p[fit]
        w=np.linalg.solve(X.T@X+1e-3*np.eye(X.shape[1]),X.T@t); out[val]+=Xv@w
    return out

def probs(tr,va,k):
    z=tr.groupby(["count","event"]).size().unstack("event",fill_value=0).reindex(index=range(12),columns=range(4),fill_value=0).to_numpy(float); prior=(z/z.sum(1,keepdims=True))[va["count"].to_numpy()]
    tab=tr.groupby(["pitcher_id","count","event"]).size().unstack("event",fill_value=0).reindex(columns=range(4),fill_value=0); ix=pd.MultiIndex.from_arrays([va.pitcher_id,va["count"]]); C=tab.reindex(ix).fillna(0).to_numpy(float); n=C.sum(1,keepdims=True)
    return (C+k*prior)/(n+k)-prior

use=["row_id","season","inning","top_bottom","balls_before","strikes_before","pitcher_id","batter_id","control_success"]
d=pd.read_csv("../data/train.csv",usecols=use,encoding="utf-8-sig"); d["row_num"]=d.row_id.str[6:].astype(int); d["count"]=d.balls_before*3+d.strikes_before; d["event"]=event_label(d)
for tag,yr in (("C",2022),("A",2024)):
    tr=d[d.season<yr]; va=d[d.season==yr]; y=va.control_success.to_numpy(float); p=pred(tag); base=score(y,p); order=va.row_num.to_numpy(); q0=cf(y,p,np.empty((len(va),0)),order); dint=score(y,q0)-base
    print(f"\n{tag} valid={yr} v66={base:.3f} intercept={dint:+.3f}")
    for k in (20.,50.,100.,200.,500.):
        q=cf(y,p,probs(tr,va,k),order); raw=score(y,q)-base
        print(f" K={k:5.0f} raw={raw:+8.3f} net={raw-dint:+8.3f}")
