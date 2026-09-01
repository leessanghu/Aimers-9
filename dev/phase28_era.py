"""시대 보정(era adjustment) 검증 — 커리어 레이트의 시대 혼합 편향이 실제로 해로운가.

가설: 리그 수준이 5년간 0.543->0.486로 0.057 하락(= 투수 개인차 0.0555와 동급).
      커리어 누적 레이트는 여러 시대가 섞여 베테랑 과대/신인 과소 평가.
검증: 커리어 raw vs 시대보정 커리어 -> 다음 시즌 '리그 대비 편차' 예측력 비교
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
seasons = sorted(df.season.unique())
league = df.groupby("season").control_success.mean()
print("리그 시즌별 성공률:"); print(league.round(4).to_string()); print()

sub = df.sort_values(["pitcher_id","row_num"])
last = sub.groupby(["pitcher_id","season"], as_index=False).last()
nb = last.asof_pitcher_n.fillna(0).to_numpy(float)
last["N_end"] = nb+1
last["S_end"] = np.round(last.asof_pitcher_success_rate.fillna(0).to_numpy(float)*nb)+last.control_success.to_numpy(float)
pN = last.pivot(index="pitcher_id",columns="season",values="N_end").reindex(columns=seasons).ffill(axis=1)
pS = last.pivot(index="pitcher_id",columns="season",values="S_end").reindex(columns=seasons).ffill(axis=1)

# 시즌별 '한 시즌만' 카운트
nS = pN.diff(axis=1); nS[seasons[0]] = pN[seasons[0]]
sS = pS.diff(axis=1); sS[seasons[0]] = pS[seasons[0]]

rows=[]
for i in range(1,len(seasons)):
    S,T = seasons[i-1], seasons[i]
    prior = [x for x in seasons if x<=S]
    n_tot = nS[prior].sum(axis=1)
    s_tot = sS[prior].sum(axis=1)
    career_raw = s_tot/n_tot.replace(0,np.nan)
    # 시대보정: 각 시즌 (그 시즌 레이트 - 그 시즌 리그) 를 투구수 가중 평균
    num = sum(nS[x]*(sS[x]/nS[x].replace(0,np.nan) - league[x]).fillna(0) for x in prior)
    career_adj = num/n_tot.replace(0,np.nan)
    n_t = nS[T]; s_t = sS[T]
    tgt_raw = s_t/n_t.replace(0,np.nan)
    d = pd.DataFrame({"career_raw":career_raw,"career_adj":career_adj,"n_tot":n_tot,
                      "tgt_raw":tgt_raw,"n_t":n_t}).dropna()
    d = d[(d.n_tot>=400)&(d.n_t>=300)]
    d["tgt_adj"] = d.tgt_raw - league[T]
    if len(d)>=40: rows.append(d)
a = pd.concat(rows)
print(f"표본 {len(a):,}\n")

def r2(cols,y):
    X=np.column_stack([np.ones(len(y))]+[np.asarray(c,float) for c in cols])
    b,_,_,_=np.linalg.lstsq(X,y,rcond=None); p=X@b
    return 1-((y-p)**2).sum()/((y-y.mean())**2).sum()

y = a.tgt_adj.to_numpy()   # 다음 시즌 '리그 대비 편차' = 진짜 실력
print("=== 다음시즌 리그대비편차 예측 (R^2) ===")
print(f"  커리어 raw          R2={r2([a.career_raw],y):.4f}")
print(f"  커리어 시대보정     R2={r2([a.career_adj],y):.4f}")
print(f"  둘 다               R2={r2([a.career_raw,a.career_adj],y):.4f}")
print()
print(f"  상관 raw={np.corrcoef(a.career_raw,y)[0,1]:+.4f}   보정={np.corrcoef(a.career_adj,y)[0,1]:+.4f}")
print(f"  두 변수 상관 = {np.corrcoef(a.career_raw,a.career_adj)[0,1]:.4f}")
