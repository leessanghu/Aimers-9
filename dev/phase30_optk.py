"""in-season 스무딩 K의 최적값 측정. 현재 K=15는 검증된 적이 없다.

최적 K = p(1-p) / Var(시즌 실력이 커리어 prior에서 벗어나는 진짜 편차)
편차 분산은 노이즈 제거해서 추정한다.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
seasons = sorted(df.season.unique())
r = float(df.control_success.mean()); base_var = r*(1-r)

sub = df.sort_values(["pitcher_id","row_num"])
last = sub.groupby(["pitcher_id","season"], as_index=False).last()
nb = last.asof_pitcher_n.fillna(0).to_numpy(float)
last["N_end"]=nb+1
last["S_end"]=np.round(last.asof_pitcher_success_rate.fillna(0).to_numpy(float)*nb)+last.control_success.to_numpy(float)
pN=last.pivot(index="pitcher_id",columns="season",values="N_end").reindex(columns=seasons).ffill(axis=1)
pS=last.pivot(index="pitcher_id",columns="season",values="S_end").reindex(columns=seasons).ffill(axis=1)
nS=pN.diff(axis=1); nS[seasons[0]]=pN[seasons[0]]
sS=pS.diff(axis=1); sS[seasons[0]]=pS[seasons[0]]

rows=[]
for i in range(1,len(seasons)):
    S=seasons[i]; prev=seasons[i-1]
    n_s, s_s = nS[S], sS[S]                       # 그 시즌 한 시즌만
    n_p, s_p = pN[prev], pS[prev]                 # 커리어 prior (직전 시즌 말 누적)
    d=pd.DataFrame({"n_s":n_s,"s_s":s_s,"n_p":n_p,"s_p":s_p}).dropna()
    d=d[(d.n_s>=150)&(d.n_p>=300)]
    if len(d)<40: continue
    d["p_s"]=d.s_s/d.n_s; d["p_p"]=d.s_p/d.n_p
    rows.append(d)
a=pd.concat(rows)
dev = (a.p_s - a.p_p).to_numpy()
n_s = a.n_s.to_numpy(); p_s = a.p_s.to_numpy(); n_p=a.n_p.to_numpy(); p_p=a.p_p.to_numpy()
noise = np.mean(p_s*(1-p_s)/n_s + p_p*(1-p_p)/n_p)
obs = dev.var()
true_var = max(obs - noise, 1e-8)
print(f"표본 {len(a):,} (투수,시즌)")
print(f"  시즌rate - 커리어prior 편차:  관측SD={np.sqrt(obs):.4f}  노이즈SD={np.sqrt(noise):.4f}  >>진짜SD={np.sqrt(true_var):.4f}<<")
print(f"\n  최적 K = p(1-p)/Var(진짜편차) = {base_var:.4f}/{true_var:.6f} = {base_var/true_var:.0f}")
print(f"  현재 사용중 K = 15")
print()
# K별로 '다음 부분'을 얼마나 잘 맞추는지 직접 평가: 시즌을 전/후반으로 갈라 전반->후반 예측
half=[]
for i in range(1,len(seasons)):
    S=seasons[i]; prev=seasons[i-1]
    cur=df[df.season==S].sort_values(["pitcher_id","row_num"])
    h=cur.groupby("pitcher_id").cumcount(); tot=cur.groupby("pitcher_id")["row_num"].transform("size")
    first=h < tot*0.5
    g1=cur[first].groupby("pitcher_id").control_success.agg(n1="count",s1="sum")
    g2=cur[~first].groupby("pitcher_id").control_success.agg(n2="count",s2="sum")
    pr=pd.DataFrame({"n_p":pN[prev],"s_p":pS[prev]})
    j=g1.join(g2,how="inner").join(pr,how="inner").dropna()
    j=j[(j.n1>=100)&(j.n2>=100)&(j.n_p>=300)]
    if len(j): half.append(j)
h=pd.concat(half)
print(f"전반->후반 검증 표본 {len(h):,}")
prior=(h.s_p/h.n_p).to_numpy(); n1=h.n1.to_numpy(); p1=(h.s1/h.n1).to_numpy()
y2=(h.s2/h.n2).to_numpy(); w2=h.n2.to_numpy()
print(f"  {'K':>6s}  가중MSE(후반 실제 대비)")
best=(None,1e9)
for K in [0,5,15,30,60,100,150,220,300,450,600,900]:
    est=(n1*p1+K*prior)/(n1+K)
    mse=float(np.average((est-y2)**2, weights=w2))
    if mse<best[1]: best=(K,mse)
    print(f"  {K:6d}  {mse:.6f}{'   <-- 최적' if K==best[0] and mse==best[1] else ''}")
print(f"\n>>> 경험적 최적 K = {best[0]}")
