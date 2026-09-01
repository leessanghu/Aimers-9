"""도메인 무관 감사. 162피처 다 무시하고 raw 48컬럼 자체의 관계/불일치를 찾는다.
가설: 컬럼 A와 B가 서로 유도 가능해 보이는데 실제로 완벽히 일치하지 않으면,
그 '어긋난 만큼'이 우리가 안 쓰고 있는 정보일 수 있다."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
print(f'전체 {len(df):,}행')

print('\n=== (1) home_win_expectancy + away_win_expectancy = 100 인가? ===')
s = df['home_win_expectancy'] + df['away_win_expectancy']
print(f'  합계 분포: min={s.min():.4f} max={s.max():.4f} mean={s.mean():.4f} std={s.std():.6f}')
print(f'  100과 다른 행 비율 = {(np.abs(s-100)>0.01).mean()*100:.2f}%')

print('\n=== (2) asof_pitcher_pitchmix_n vs asof_pitcher_n : 항상 같은가? ===')
d = (df['asof_pitcher_pitchmix_n'] - df['asof_pitcher_n'])
print(f'  차이 분포: min={d.min()} max={d.max()} mean={d.mean():.3f}')
print(f'  다른 행 비율 = {(d != 0).mean()*100:.2f}%')
print(d.value_counts().head(10))

print('\n=== (3) score_diff_home vs score_diff_pitcher_team vs top_bottom ===')
# top_bottom='T'(초,원정공격,홈팀투수) 가정 검증
for tb in df['top_bottom'].unique():
    sub = df[df['top_bottom'] == tb]
    same = (sub['score_diff_home'] == sub['score_diff_pitcher_team']).mean()
    opp = (sub['score_diff_home'] == -sub['score_diff_pitcher_team']).mean()
    print(f'  top_bottom={tb}: score_diff_home==score_diff_pitcher_team 비율={same*100:.1f}%  '
          f'==-1배 비율={opp*100:.1f}%')

print('\n=== (4) run_total_before = run_top_before + run_bot_before 인가? ===')
diff = df['run_total_before'] - (df['run_top_before'] + df['run_bot_before'])
print(f'  불일치 비율 = {(diff != 0).mean()*100:.4f}%')

print('\n=== (5) base_state 문자열이 runner_on_1b/2b/3b와 완전히 같은 정보인가? ===')
expect = df['runner_on_1b'].astype(str) + df['runner_on_2b'].astype(str) + df['runner_on_3b'].astype(str)
bs = df['base_state'].astype(str).str.replace('_', '0')
mismatch = (expect != bs).mean()
print(f'  불일치 비율 = {mismatch*100:.2f}%')
if mismatch > 0:
    print(df.loc[expect != bs, ['base_state','runner_on_1b','runner_on_2b','runner_on_3b']].head(5))

print('\n=== (6) game_type 값 종류 및 비율 ===')
print(df['game_type'].value_counts())
print('  season별 game_type 분포:')
print(pd.crosstab(df['season'], df['game_type']))

print('\n=== (7) li(leverage index)가 score_diff/base_state/outs만으로 완전히 결정되는가? ===')
# 같은 (inning, outs, base_state, score_diff_home)인데 li가 다른 경우가 있으면
# li에 그 조합만으로 설명 안 되는 추가정보(팀전력 등)가 있다는 뜻
key = list(zip(df['inning'], df['outs_before'], df['base_state'], df['score_diff_home'].clip(-5,5)))
tmp = pd.DataFrame({'key': key, 'li': df['li']})
grp = tmp.groupby('key')['li'].agg(['std', 'count'])
grp = grp[grp['count'] >= 50]
print(f'  동일 (inning,outs,base_state,score_diff) 조합 {len(grp)}개(표본50+)')
print(f'  그 안에서 li 표준편차 분포: mean={grp["std"].mean():.4f} median={grp["std"].median():.4f} max={grp["std"].max():.4f}')
print('  -> 0에 가까우면 li=상황만으로 결정(새정보없음), 크면 li에 팀전력차 등 추가정보 있음')

print('\n=== (8) pitcher_team_id, batter_team_id 조합(매치업) 자체를 써본 적 있나 ===')
print(f'  고유 pitcher_team_id 수 = {df["pitcher_team_id"].nunique()}')
print(f'  고유 batter_team_id 수 = {df["batter_team_id"].nunique()}')
print(f'  고유 (pitcher_team,batter_team) 조합 수 = {df.groupby(["pitcher_team_id","batter_team_id"]).ngroups}')
same_team = (df['pitcher_team_id'] == df['batter_team_id']).mean()
print(f'  pitcher_team_id == batter_team_id 인 행 비율 = {same_team*100:.4f}%  (0이어야 정상, 아니면 이상데이터)')
