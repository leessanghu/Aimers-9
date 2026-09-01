"""시즌별 control_success 하락(0.5647->0.4861)의 원인 측정.
측정 가능한 모든 각도:
 1) 행별 결과유형 복원 -> 정확히 뭐가 늘었나 (reverse? middle? ball?)
 2) 같은 투수 연도별 변화 (개인 하락 vs 선수구성 변화)
 3) 구성 변화: 역할(선발/불펜), F/R, 카운트 분포
 4) trackman 물리량: 구속/회전/무브먼트/구종 추세
"""
import numpy as np, pandas as pd, sys, time
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
def log(m): print(f'[{time.time()-t0:5.0f}s] {m}', flush=True)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'season', 'pitcher_id', 'game_type', 'inning',
                          'balls_before', 'strikes_before', 'asof_pitcher_n',
                          'asof_pitcher_success_rate', 'asof_pitcher_reverse_rate',
                          'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate',
                          'asof_pitcher_strike_rate', 'control_success'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)
log(f'로드 {len(df):,}')

# ---------- 1) 행별 결과유형 복원 ----------
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
def cnt(col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
S_, R_, M_, B_, K_ = [cnt(c) for c in ['asof_pitcher_success_rate', 'asof_pitcher_reverse_rate',
                                        'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate',
                                        'asof_pitcher_strike_rate']]
ordr = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
pid_o = df['pitcher_id'].to_numpy()[ordr]
n_o = n_[ordr]
hstep = np.zeros(len(df), dtype=bool)
hstep[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
diffs = {}
for nm, arr in [('rev', R_), ('mid', M_), ('ball', B_), ('strike', K_), ('succ', S_)]:
    d = np.zeros(len(df)); d[ordr[:-1]] = np.diff(arr[ordr])
    diffs[nm] = d
log(f'복원 가능 행: {hstep.sum():,} ({hstep.mean()*100:.1f}%)')

sub = df.loc[hstep].copy()
for nm in diffs:
    sub[nm] = (diffs[nm][hstep] > 0).astype(int)
print()
print('=== 1) 결과유형 구성비 (시즌별, 복원행 기준) ===')
comp = sub.groupby('season')[['succ', 'rev', 'mid', 'ball', 'strike']].mean()
print(comp.round(4).to_string())
print()
print('전년대비 변화:')
print(comp.diff().round(4).to_string())
print()
print('2019->2024 총변화:')
print((comp.loc[2024] - comp.loc[2019]).round(4).to_string())
print()

# ---------- 2) 같은 투수 연도별 (개인 하락 vs 구성 변화) ----------
print('=== 2) 같은 투수 연속연도 비교 (개인 내 변화) ===')
py = df.groupby(['pitcher_id', 'season'])['control_success'].agg(['mean', 'count']).reset_index()
py = py[py['count'] >= 200]
rows = []
for s in range(2020, 2025):
    a = py[py.season == s - 1].set_index('pitcher_id')
    b = py[py.season == s].set_index('pitcher_id')
    common = a.index.intersection(b.index)
    if len(common) == 0:
        continue
    delta = (b.loc[common, 'mean'] - a.loc[common, 'mean'])
    # 전체 리그 변화
    league = df[df.season == s]['control_success'].mean() - df[df.season == s - 1]['control_success'].mean()
    rows.append((s, len(common), delta.mean(), league))
print(f'{"연도":>6s} {"공통투수":>8s} {"개인평균변화":>12s} {"리그전체변화":>12s} {"구성효과":>10s}')
for s, n, d, l in rows:
    print(f'{s:>6d} {n:>8d} {d:>+12.4f} {l:>+12.4f} {l-d:>+10.4f}')
print()
print('※ 개인평균변화 ≈ 리그전체변화 이면 "개별 투수가 실제로 나빠짐"')
print('   구성효과가 크면 "선수 구성이 바뀌어서 평균이 내려감"')
print()

# ---------- 3) 구성 변화 ----------
print('=== 3) 구성 변화 ===')
print('game_type별 성공률:')
gt = df.groupby(['season', 'game_type'])['control_success'].mean().unstack()
print(gt.round(4).to_string())
print()
print('카운트 분포 변화 (0-0 비율, 2스트라이크 비율):')
df['is00'] = ((df.balls_before == 0) & (df.strikes_before == 0)).astype(int)
df['is2k'] = (df.strikes_before == 2).astype(int)
df['is3b'] = (df.balls_before == 3).astype(int)
print(df.groupby('season')[['is00', 'is2k', 'is3b']].mean().round(4).to_string())
print()
print('투수당 시즌 투구수(경험 분포):')
pn = df.groupby(['season', 'pitcher_id']).size().groupby('season').agg(['mean', 'median', 'count'])
print(pn.round(1).to_string())
log('train 분석 완료')
