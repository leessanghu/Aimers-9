"""투수 실력 추정 정밀도 진단 — 남은 격차가 여기 있는지 확인.

배경: 투수 실력 진짜SD=0.0555 -> BSS 상한 1235점. 1등이 1200점.
      즉 상위권은 실력 추정을 거의 완벽히 하고 있고, 우리(963)의 격차는
      '새 피처 종류'가 아니라 '실력 추정 정밀도'일 가능성이 크다.

측정 1: 작년 한 시즌만 vs 커리어 누적 — 다음 시즌 예측력 비교
측정 2: success 외 4개 rate(ball/strike/middle/reverse)가 예측력을 더하는가
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_","",regex=False).astype(int)
g = float(df.control_success.mean())
seasons = sorted(df.season.unique())

# 시즌별 (투수) 집계 + 시즌 말 누적 rate 복원용
sub = df.sort_values(["pitcher_id","row_num"])
last = sub.groupby(["pitcher_id","season"], as_index=False).last()
nb = last.asof_pitcher_n.fillna(0).to_numpy(float)
last["N_end"] = nb + 1
last["S_end"] = np.round(last.asof_pitcher_success_rate.fillna(0).to_numpy(float)*nb) + last.control_success.to_numpy(float)
for c,nm in [("asof_pitcher_ball_rate","B"),("asof_pitcher_strike_rate","K"),
             ("asof_pitcher_middle_rate","M"),("asof_pitcher_reverse_rate","R")]:
    last[f"{nm}_end"] = np.round(last[c].fillna(0).to_numpy(float)*nb)
piv = {c: last.pivot(index="pitcher_id", columns="season", values=c).reindex(columns=seasons).ffill(axis=1)
       for c in ["N_end","S_end","B_end","K_end","M_end","R_end"]}

# 그 시즌 '한 시즌만' = 누적(S) - 누적(S-1)
rows=[]
for i in range(1,len(seasons)):
    S, T = seasons[i-1], seasons[i]
    prev = seasons[i-2] if i>=2 else None
    N_c, S_c = piv["N_end"][S], piv["S_end"][S]           # 커리어 누적(S까지)
    if prev is not None:
        N_l = piv["N_end"][S]-piv["N_end"][prev]; S_l = piv["S_end"][S]-piv["S_end"][prev]
        B_l = piv["B_end"][S]-piv["B_end"][prev]; K_l = piv["K_end"][S]-piv["K_end"][prev]
        M_l = piv["M_end"][S]-piv["M_end"][prev]; R_l = piv["R_end"][S]-piv["R_end"][prev]
    else:
        N_l,S_l,B_l,K_l,M_l,R_l = N_c,S_c,piv["B_end"][S],piv["K_end"][S],piv["M_end"][S],piv["R_end"][S]
    # 타깃: 다음 시즌 한 시즌만
    N_t = piv["N_end"][T]-piv["N_end"][S]; S_t = piv["S_end"][T]-piv["S_end"][S]
    d = pd.DataFrame({"N_c":N_c,"S_c":S_c,"N_l":N_l,"S_l":S_l,"B_l":B_l,"K_l":K_l,"M_l":M_l,"R_l":R_l,
                      "N_t":N_t,"S_t":S_t}).dropna()
    d = d[(d.N_c>=400)&(d.N_l>=300)&(d.N_t>=300)]
    if len(d)<40: continue
    d["career"]=d.S_c/d.N_c; d["lastyr"]=d.S_l/d.N_l; d["tgt"]=d.S_t/d.N_t
    for nm,col in [("ball","B_l"),("strike","K_l"),("middle","M_l"),("rev","R_l")]:
        d[nm]=d[col]/d.N_l
    rows.append(d[["career","lastyr","ball","strike","middle","rev","tgt","N_l","N_t"]])
a = pd.concat(rows)
print(f"표본 {len(a):,} (투수,시즌쌍)\n")

def r2(cols, y):
    n = len(y)
    X = np.column_stack([np.ones(n)] + [np.asarray(c, float) for c in cols])
    beta,_,_,_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X@beta
    return 1 - ((y-pred)**2).sum()/((y-y.mean())**2).sum()

y = a.tgt.to_numpy()
print("=== 측정1: 무엇으로 다음 시즌을 예측하나 (R^2) ===")
print(f"  커리어 누적만          R2={r2([a.career],y):.4f}")
print(f"  작년 한 시즌만         R2={r2([a.lastyr],y):.4f}")
print(f"  둘 다                  R2={r2([a.career,a.lastyr],y):.4f}")
print()
print("=== 측정2: 다른 rate 4개를 더하면 (작년 기준) ===")
print(f"  작년 success만         R2={r2([a.lastyr],y):.4f}")
print(f"  + ball,strike          R2={r2([a.lastyr,a.ball,a.strike],y):.4f}")
print(f"  + middle,reverse       R2={r2([a.lastyr,a.middle,a.rev],y):.4f}")
print(f"  + 4개 전부             R2={r2([a.lastyr,a.ball,a.strike,a.middle,a.rev],y):.4f}")
print(f"  커리어+작년+4개 전부   R2={r2([a.career,a.lastyr,a.ball,a.strike,a.middle,a.rev],y):.4f}")
print()
print("=== 참고: 상관 ===")
for c in ["career","lastyr","ball","strike","middle","rev"]:
    print(f"  {c:8s} vs 다음시즌  r={np.corrcoef(a[c],y)[0,1]:+.4f}")
