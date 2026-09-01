"""매칭 정밀도 검증 + 투수x구종 제구력의 진짜 신호 크기 측정.

정밀도: batter_hand는 매칭키에 안 썼으므로 독립 검증 변수로 사용.
신호크기: 우리 기준 — platoon(진짜SD 0.0438, 재현상관 0.328) 성공 / r=0.05~0.12대 전부 실패.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

m = pd.read_csv("pitcher_map.csv").sort_values("sim",ascending=False).drop_duplicates("tm_id")
p2t = m.set_index("pitcher_id")["tm_id"]
tr = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
    usecols=["row_id","season","game_month","game_dayofweek","inning","top_bottom","balls_before",
             "strikes_before","outs_before","pitcher_id","batter_hand","control_success"])
tr["tm_id"]=tr.pitcher_id.map(p2t); tr=tr.dropna(subset=["tm_id"]); tr["tm_id"]=tr.tm_id.astype(int)
tr["_tb"]=tr.top_bottom.astype(str).map({"T":"Top","B":"Bottom"})
tm = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig",
    usecols=["season","game_month","game_dayofweek","inning","top_bottom","balls_before",
             "strikes_before","outs_before","pitcher_trackman_id","pitch_type_group","batter_hand"])
tm=tm.rename(columns={"pitcher_trackman_id":"tm_id","batter_hand":"tm_bhand"})
tm=tm[tm.tm_id.isin(set(tr.tm_id))]; tm["_tb"]=tm.top_bottom.astype(str)
KEY=["season","game_month","game_dayofweek","tm_id","inning","_tb","balls_before","strikes_before","outs_before"]

# 셀 안에서 구종이 단일이고, 타자손도 단일인 경우만 확정 매칭
agg = tm.groupby(KEY).agg(n_type=("pitch_type_group","nunique"),
                          ptype=("pitch_type_group","first"),
                          n_bh=("tm_bhand","nunique"),
                          bh=("tm_bhand","first"))
j = tr.join(agg, on=KEY)
det = j[j.n_type==1].copy()
print(f"확정 매칭 {len(det):,}행 (매핑투수 대비 {len(det)/len(tr)*100:.1f}%)\n")

print("=== 정밀도 검증: batter_hand (매칭키 미사용, 독립변수) ===")
chk = det[det.n_bh==1].copy()
print(f"  타자손 비교 가능 {len(chk):,}행")
print("  train batter_hand 값:", sorted(chk.batter_hand.unique()), " tm:", sorted(chk.bh.astype(str).unique()))
for a,b in [(1,"Left"),(2,"Right")]:
    sub=chk[chk.batter_hand==a]
    if len(sub): print(f"    train={a} -> tm 분포: {sub.bh.value_counts(normalize=True).round(3).to_dict()}")

print("\n=== 투수 x 구종 제구력 신호 크기 ===")
print("  구종 분포:", det.ptype.value_counts(normalize=True).round(3).to_dict())
g_all = det.control_success.mean(); base_var=g_all*(1-g_all)
mp = det.groupby("pitcher_id").control_success.mean()
mt = det.groupby("ptype").control_success.mean()
print(f"  구종별 전역 성공률: {mt.round(4).to_dict()}")
cell = det.groupby(["pitcher_id","ptype"]).control_success.agg(["mean","count"])
cell = cell[cell["count"]>=100].reset_index()
resid = cell["mean"] - cell.pitcher_id.map(mp) - cell.ptype.map(mt) + g_all
noise = np.mean(cell["mean"]*(1-cell["mean"])/cell["count"])
tv = max(resid.var()-noise, 0.0)
print(f"  cells={len(cell):,}  관측SD={resid.std():.4f}  노이즈SD={np.sqrt(noise):.4f}  >>진짜SD={np.sqrt(tv):.4f}<<")
print(f"  상한 ~{tv/base_var*100000:.0f}점")

print("\n=== 재현성 (직전시즌 편차 -> 다음시즌 같은 편차) ===")
seasons=sorted(det.season.unique())
c2 = det.groupby(["pitcher_id","ptype","season"]).control_success.agg(s="sum",n="count")
m2 = det.groupby(["pitcher_id","season"]).control_success.agg(s="sum",n="count")
DP,DN=[],[]
for i in range(1,len(seasons)):
    S,T=seasons[i-1],seasons[i]; pr=[x for x in seasons if x<=S]
    cp=c2[c2.index.get_level_values("season").isin(pr)].groupby(level=[0,1]).sum()
    mp_=m2[m2.index.get_level_values("season").isin(pr)].groupby(level=0).sum()
    cn=c2[c2.index.get_level_values("season")==T].droplevel("season")
    mn=m2[m2.index.get_level_values("season")==T].droplevel("season")
    jj=cp.join(cn,how="inner",lsuffix="_p",rsuffix="_n")
    jj=jj[(jj.n_p>=100)&(jj.n_n>=50)]
    if not len(jj): continue
    k=jj.index.get_level_values(0)
    a1,b1=mp_["s"].reindex(k).to_numpy(),mp_["n"].reindex(k).to_numpy()
    a2,b2=mn["s"].reindex(k).to_numpy(),mn["n"].reindex(k).to_numpy()
    ok=(b1>0)&(b2>0)&~np.isnan(a1)&~np.isnan(a2)
    if ok.sum()<30: continue
    DP.append((jj.s_p.to_numpy()/jj.n_p.to_numpy())[ok]-(a1/b1)[ok])
    DN.append((jj.s_n.to_numpy()/jj.n_n.to_numpy())[ok]-(a2/b2)[ok])
dp,dn=np.concatenate(DP),np.concatenate(DN)
r=np.corrcoef(dp,dn)[0,1]
rng=np.random.default_rng(0)
bs=[np.corrcoef(dp[i],dn[i])[0,1] for i in (rng.integers(0,len(dp),len(dp)) for _ in range(400))]
lo,hi=np.percentile(bs,[2.5,97.5])
print(f"  n={len(dp):,}  재현상관 r={r:+.4f}  95%CI=[{lo:+.4f},{hi:+.4f}]")
print(f"\n  [기준] platoon r=+0.328 성공 / 볼카운트 +0.121·월 +0.079·상대팀 +0.054 전부 실패")
