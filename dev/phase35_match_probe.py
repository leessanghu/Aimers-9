"""train <-> Trackman 투구 단위 매칭 가능성 진단 (구축 전 타당성 확인).

핵심 질문: 상태키만으로 train 행이 Trackman 투구에 '유일하게' 대응되는 비율은?
유일 매칭률이 낮으면 P(success|투수,구종) 추정 자체가 불가능하므로 전체 아이디어가 무의미.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd

m = pd.read_csv("pitcher_map.csv").sort_values("sim",ascending=False).drop_duplicates("tm_id")
p2t = m.set_index("pitcher_id")["tm_id"]
print(f"매핑된 투수 {len(p2t)}명")

tr = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
    usecols=["row_id","season","game_month","game_dayofweek","inning","top_bottom",
             "balls_before","strikes_before","outs_before","pitcher_id","pitcher_team_id","batter_team_id"])
tr["tm_id"] = tr.pitcher_id.map(p2t)
cov = tr.tm_id.notna().mean()
print(f"train 행 중 투수 매핑된 비율: {cov:.3f}  ({tr.tm_id.notna().sum():,}행)")
tr = tr.dropna(subset=["tm_id"]); tr["tm_id"]=tr.tm_id.astype(int)

tm = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig",
    usecols=["season","game_month","game_dayofweek","inning","top_bottom","balls_before",
             "strikes_before","outs_before","pitcher_trackman_id","pitch_type_group","trackman_game_id"])
tm = tm.rename(columns={"pitcher_trackman_id":"tm_id"})
tm = tm[tm.tm_id.isin(set(tr.tm_id))]
print(f"Trackman 투구 중 매핑 투수 것: {len(tm):,}\n")

# top_bottom 표기 확인
print("top_bottom 값:", "train", sorted(tr.top_bottom.astype(str).unique())[:5],
      "| tm", sorted(tm.top_bottom.astype(str).unique())[:5])
tb_map = {"T":"Top","B":"Bottom"}
tr["_tb"] = tr.top_bottom.astype(str).map(lambda v: tb_map.get(v, v))
tm["_tb"] = tm.top_bottom.astype(str)
common = set(tr._tb.unique()) & set(tm._tb.unique())
print("공통 top_bottom 값:", common, "\n")

KEY = ["season","game_month","game_dayofweek","tm_id","inning","_tb",
       "balls_before","strikes_before","outs_before"]
cnt = tm.groupby(KEY).size().rename("n_cand")
j = tr.join(cnt, on=KEY)
j["n_cand"] = j.n_cand.fillna(0)
tot = len(j)
print("=== 상태키 매칭 결과 (매핑 투수 행 기준) ===")
for lab, msk in [("후보 0개 (매칭 실패)", j.n_cand==0),
                 ("후보 1개 (유일 매칭)", j.n_cand==1),
                 ("후보 2개", j.n_cand==2),
                 ("후보 3~5개", (j.n_cand>=3)&(j.n_cand<=5)),
                 ("후보 6개 이상", j.n_cand>=6)]:
    print(f"  {lab:22s} {msk.sum():9,}  ({msk.mean()*100:5.1f}%)")
uniq = (j.n_cand==1).mean()
print(f"\n전체 train 대비 유일매칭 커버리지 = {uniq*cov*100:.1f}%")

# 후보가 여럿이어도 구종이 하나면 사실상 확정
tmg = tm.groupby(KEY).pitch_type_group.nunique().rename("n_type")
j2 = tr.join(tmg, on=KEY); j2["n_type"]=j2.n_type.fillna(0)
det = (j2.n_type==1)
print(f"후보 여럿이라도 '구종 단일'이라 확정 가능: {det.mean()*100:5.1f}%  "
      f"(전체 train 대비 {det.mean()*cov*100:.1f}%)")
print(f"\n>>> 채택 기준(커버리지 50%) 대비: {det.mean()*cov*100:.1f}%")
