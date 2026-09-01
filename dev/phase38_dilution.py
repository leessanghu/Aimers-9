"""구종 피처가 +6.7밖에 못 낸 이유 규명: pt_dev 분산의 분해.

pt_dev = sum_t P(t|투수,카운트) * ctrl(투수,t) - prior(투수)
  - 투수간(between) 성분: 그 투수의 평균 믹스 x 구종별 실력 -> prior와 상당부분 중복
  - 투수내(within) 성분: 카운트에 따라 믹스가 흔들려 생기는 변동 -> 순수 신규 신호
후자가 작으면 이 축은 이미 고갈. 크면 P(t|x) 모델링을 개선할 여지가 있다.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from inseason import build_season_end_table, _pivots_from_table
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df.row_id.str.replace("TRAIN_","",regex=False).astype(int)
g = float(df.control_success.mean()); sr = sorted(df.season.unique().tolist())
se = build_season_end_table(df); piv=_pivots_from_table(se,sr)
idx = pd.MultiIndex.from_arrays([df.pitcher_id, df.season-1])
pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
mt = build_matched(df); tb = build_pitchtype_tables(mt, sr)
f = transform_pitchtype(df, tb, pp, g, sr)

d = f.pt_dev.to_numpy()
pid = df.pitcher_id.to_numpy()
s = pd.Series(d).groupby(pid).transform("mean").to_numpy()
within = d - s
print(f"pt_dev 전체 SD      = {d.std():.5f}")
print(f"  투수간(between) SD = {s.std():.5f}   분산비 {s.var()/d.var()*100:5.1f}%")
print(f"  투수내(within)  SD = {within.std():.5f}   분산비 {within.var()/d.var()*100:5.1f}%")
print()
print("[해석] between 성분은 '이 투수의 평균적 구종구성 x 구종별 실력'이라")
print("       prior(투수 전체 실력)와 상당부분 중복. within 성분만이 순수 신규 신호.")
print()
print("[기준] 우리 성공 피처 진폭: platoon 0.0119 / inning 0.0093 / 실패구간 0.0044~0.0065")
print(f">>> 구종의 순수 신규 신호(within) = {within.std():.5f}")
