"""PA-event 차분 라벨을 row-independent 과거표로 예측하는 빠른 파일럿.

train<=2023에서 P(event | count, pitcher/batter)와 P(type | count,pitcher)를 만들고
2024 각 행만 조회한다. test 행간 참조 없이 동일하게 구현 가능한 구조다.
패키징은 하지 않는다.
"""

import sys
import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def build_event(df):
    d = df.sort_values("row_num").reset_index()
    same = np.r_[
        (d.pitcher_id.to_numpy()[1:] == d.pitcher_id.to_numpy()[:-1])
        & (d.batter_id.to_numpy()[1:] == d.batter_id.to_numpy()[:-1])
        & (d.inning.to_numpy()[1:] == d.inning.to_numpy()[:-1])
        & (d.top_bottom.to_numpy()[1:] == d.top_bottom.to_numpy()[:-1]), False]
    b = d.balls_before.to_numpy(); s = d.strikes_before.to_numpy()
    bn = np.r_[b[1:], -99]; sn = np.r_[s[1:], -99]
    e = np.full(len(d), 3, dtype=np.int8)
    e[same & (bn == b+1) & (sn == s)] = 0
    e[same & (sn == s+1) & (bn == b)] = 1
    e[same & (bn == b) & (sn == s) & (s == 2)] = 2
    out = np.empty(len(d), dtype=np.int8); out[d["index"].to_numpy()] = e
    return out


def v66_valid():
    avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
    base = avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6", "d8", "sub")])
    hur = np.mean([(1-np.load(f"phase90_cache/A_core_{n}.npy"))*np.load(f"phase90_cache/A_snc_{n}.npy") for n in ("d6", "d8")], axis=0)
    mr = avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42,7)])
    od = avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42,7)])
    mo = avg([f"idea46_cache/A_midother_s{s}.npy" for s in (42,7)])
    cb = avg([f"idea54_cache/A_cond_ball_s{s}.npy" for s in (42,7)])
    cr = avg([f"idea54_cache/A_count_resid_s{s}.npy" for s in (42,7)])
    f5 = avg([f"idea54_cache/A_future50_multi_s{s}.npy" for s in (42,7)])
    return .1824*base+.2432*hur+.0608*mr+.1216*od+.1520*mo+.08*cb+.08*cr+.08*f5


def score(y, p):
    p = np.clip(p, 0, 1)
    return 1e5*(1-np.mean((p-y)**2)/(y.mean()*(1-y.mean())))


def probs_by_entity(train, valid, entity, label, nclass, prior, k):
    """(entity,count) 다항 카운트를 count prior로 평활."""
    tab = train.groupby([entity, "count", label]).size().unstack(label, fill_value=0)
    tab = tab.reindex(columns=range(nclass), fill_value=0)
    idx = pd.MultiIndex.from_arrays([valid[entity], valid["count"]])
    C = tab.reindex(idx).fillna(0).to_numpy(float)
    n = C.sum(axis=1, keepdims=True)
    return (C + k*prior)/(n+k)


def type_probs(train, valid, prior, k):
    tab = train.groupby(["pitcher_id", "count", "ptype"]).size().unstack("ptype", fill_value=0).reindex(columns=range(3), fill_value=0)
    idx = pd.MultiIndex.from_arrays([valid.pitcher_id, valid["count"]])
    C = tab.reindex(idx).fillna(0).to_numpy(float); n=C.sum(axis=1,keepdims=True)
    return (C+k*prior)/(n+k)


use = ["row_id","season","inning","top_bottom","balls_before","strikes_before","pitcher_id","batter_id",
       "asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate","control_success"]
df = pd.read_csv("../data/train.csv", usecols=use, encoding="utf-8-sig")
df["row_num"] = df.row_id.str.replace("TRAIN_", "", regex=False).astype(int)
df["count"] = df.balls_before*3+df.strikes_before
df["event"] = build_event(df)
df["ptype"] = np.load("pitchtype_labels.npy").astype(int)
tr = df[(df.season<=2023)&(df.ptype>=0)].copy(); va=df[df.season==2024].copy()
y=va.control_success.to_numpy(float); v66=v66_valid(); base=score(y,v66)
print(f"v66={base:.3f} train={len(tr):,} valid={len(va):,}")

# count별 prior P(event), P(type)
def count_prior(nclass, label):
    z=tr.groupby(["count",label]).size().unstack(label,fill_value=0).reindex(index=range(12),columns=range(nclass),fill_value=0)
    a=z.to_numpy(float); return a/a.sum(axis=1,keepdims=True)

pe0=count_prior(4,"event")[va["count"].to_numpy()]
pt0=count_prior(3,"ptype")[va["count"].to_numpy()]
mix=va[["asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]].fillna(0).to_numpy(float)
mix=np.divide(mix,mix.sum(axis=1,keepdims=True),out=pt0.copy(),where=mix.sum(axis=1,keepdims=True)>0)

# E[y|event,count]와 E[y|event,count,type], <=2023 고정
mu0=float(tr.control_success.mean())
mue=tr.groupby(["count","event"]).control_success.mean()
muept=tr.groupby(["count","event","ptype"]).control_success.mean()
mu_e=np.full((12,4),mu0); mu_ept=np.full((12,4,3),mu0)
for (c,e),z in mue.items(): mu_e[int(c),int(e)]=z
for (c,e,t),z in muept.items(): mu_ept[int(c),int(e),int(t)]=z
cnt=va["count"].to_numpy()

preds={}
for kp in (50.,200.,1000.):
    pp=probs_by_entity(tr,va,"pitcher_id","event",4,pe0,kp)
    for kb in (50.,200.,1000.):
        pb=probs_by_entity(tr,va,"batter_id","event",4,pe0,kb)
        for wb in (0.,.25,.5,.75,1.):
            pe=(1-wb)*pp+wb*pb
            name=f"event_kp{int(kp)}_kb{int(kb)}_wb{wb:.2f}"
            preds[name]=np.sum(pe*mu_e[cnt],axis=1)

# event + pitchtype 독립 주변화. P(type)는 공식 career mix 또는 pitcher×count table.
best_event=max(preds, key=lambda n: score(y,preds[n]))
kp=int(best_event.split("kp")[1].split("_")[0]); kb=int(best_event.split("kb")[1].split("_")[0]); wb=float(best_event.split("wb")[1])
pe=(1-wb)*probs_by_entity(tr,va,"pitcher_id","event",4,pe0,kp)+wb*probs_by_entity(tr,va,"batter_id","event",4,pe0,kb)
for kt in (50.,200.,1000.):
    pth=type_probs(tr,va,pt0,kt)
    for wt,namept in ((0.,"career"),(0.5,"half"),(1.,"histcount")):
        ptp=(1-wt)*mix+wt*pth
        pred=np.einsum("ne,nt,net->n",pe,ptp,mu_ept[cnt])
        preds[f"eventxtype_{namept}_kt{int(kt)}"]=pred

rows=[]
for name,p in preds.items():
    solo=score(y,p); corr=np.corrcoef(p,v66)[0,1]
    bestd=-1e9; bestw=0
    for w in np.linspace(0,0.30,121):
        d=score(y,(1-w)*v66+w*p)-base
        if d>bestd: bestd,bestw=d,w
    rows.append((name,solo,corr,bestd,bestw))
out=pd.DataFrame(rows,columns=["name","solo","corr_v66","blend_delta","blend_w"]).sort_values("blend_delta",ascending=False)
print(out.head(20).to_string(index=False))
out.to_csv("idea74c_event_table_results.csv",index=False)
