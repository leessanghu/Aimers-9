"""raw 포렌식 5차 — 아직 안 쓴 축 탐색.

핵심 아이디어: asof_pitcher_prev1_game_success_rate 는 '직전 경기'의 성적이므로
같은 투수 안에서 이 값이 바뀌는 순간 = 새 경기 시작(경기 경계).
경계를 알면 '이번 등판 N번째 투구'(등판내 피로도)를 복원할 수 있다.
우리 162피처엔 등판내 투구수가 전혀 없다(asof_pitcher_n=커리어누적, inseason_n=시즌누적,
role_ppa=커리어평균 등판당투구수. 전부 '지금 이 등판에서 몇 개째'가 아님).

확인할 것:
 (1) 경기 경계가 실제로 복원되는가 (경계 간격이 등판당 투구수로 말이 되는가)
 (2) 등판내 투구수별 성공률 (피로 효과가 실재하는가)
 (3) 그게 inning으로 이미 설명되는가 (=새 정보가 아닌가)  <- 이게 결정적
 (4) 덤: 직전 투구에서 득점이 났는가 (run_total_before 차분)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
y = df['control_success'].to_numpy(np.float64)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
inning = df['inning'].to_numpy()
g = y.mean()

o = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_prev = np.zeros(len(o), dtype=bool)
same_prev[1:] = (pid[o][1:] == pid[o][:-1])

print(f'전역 성공률={g:.4f}\n')
print('=' * 78)
print('[1] 경기 경계 복원 (prev1_game_success_rate 변화 시점)')
print('=' * 78)
p1 = df['asof_pitcher_prev1_game_success_rate'].fillna(-1).to_numpy(np.float64)
p1o = p1[o]
changed = np.zeros(len(o), dtype=bool)
changed[1:] = (np.abs(p1o[1:] - p1o[:-1]) > 1e-12) & same_prev[1:]
new_game_ord = (~same_prev) | changed          # 투수가 바뀌거나 prev1이 바뀌면 새 경기
print(f'  새 경기 시작 지점: {new_game_ord.sum():,} ({new_game_ord.mean()*100:.2f}%)')

# 등판내 투구순번
outing_idx_ord = np.zeros(len(o), dtype=np.int64)
cur = 0
for i in range(len(o)):
    if new_game_ord[i]:
        cur = 0
    outing_idx_ord[i] = cur
    cur += 1
outing_idx = np.empty(n, np.int64); outing_idx[o] = outing_idx_ord

# 등판별 총 투구수 분포
seg_id_ord = np.cumsum(new_game_ord.astype(np.int64))
seg_len = pd.Series(seg_id_ord).value_counts()
print(f'  등판당 투구수: 중앙값={seg_len.median():.0f}  '
      f'p25={seg_len.quantile(.25):.0f}  p75={seg_len.quantile(.75):.0f}  '
      f'p95={seg_len.quantile(.95):.0f}  최대={seg_len.max()}')
print(f'  (실제 야구: 선발 80~110개, 불펜 10~25개 -> 이 분포가 말이 되는지 확인)')

print('\n' + '=' * 78)
print('[2] 등판내 투구순번별 성공률 (피로 효과)')
print('=' * 78)
bins = [(0, 4), (5, 14), (15, 29), (30, 49), (50, 74), (75, 99), (100, 10000)]
for lo, hi in bins:
    m = (outing_idx >= lo) & (outing_idx <= hi)
    if m.sum() < 2000:
        continue
    print(f'  {lo:>3}-{hi if hi<10000 else "+":<5} n={m.sum():>9,} ({m.mean()*100:5.2f}%)  '
          f'성공률={y[m].mean():.4f}  편차={y[m].mean()-g:+.4f}')

print('\n' + '=' * 78)
print('[3] inning으로 이미 설명되는가? (등판순번 vs inning 상관 / inning 통제 후 잔여효과)')
print('=' * 78)
corr = np.corrcoef(outing_idx, inning)[0, 1]
print(f'  corr(등판내순번, inning) = {corr:+.4f}')
print(f'\n  inning 고정했을 때 등판내순번별 성공률 (같은 이닝 안에서도 차이 나는가):')
print(f'{"inning":>7}{"순번0-9":>12}{"순번10-29":>12}{"순번30-59":>12}{"순번60+":>12}')
for inn in (1, 3, 5, 7):
    row = [f'{inn:>7}']
    for lo, hi in [(0, 9), (10, 29), (30, 59), (60, 10000)]:
        m = (inning == inn) & (outing_idx >= lo) & (outing_idx <= hi)
        row.append(f'{y[m].mean():.4f}({m.sum()//1000}k)' if m.sum() > 1500 else '  -  ')
    print(''.join(f'{c:>12}' for c in row[1:]).rjust(0) and f'{inn:>7}' + ''.join(f'{c:>12}' for c in row[1:]))

print('\n' + '=' * 78)
print('[4] 덤: 직전 투구에서 득점 발생 여부')
print('=' * 78)
rt = df['run_total_before'].to_numpy(np.float64)
rto = rt[o]
drun = np.full(len(o), np.nan)
drun[1:] = rto[1:] - rto[:-1]
drun[~same_prev] = np.nan
scored = np.full(n, np.nan); scored[o] = drun
ok = np.isfinite(scored)
print(f'  복원 가능: {ok.sum():,} ({ok.mean()*100:.1f}%)')
for v, nm in [(0, '직전 무득점'), (1, '직전 1점'), (2, '직전 2점+')]:
    m = ok & ((scored == v) if v < 2 else (scored >= 2))
    if m.sum() < 1000:
        continue
    print(f'  {nm:<12} n={m.sum():>9,}  성공률={y[m].mean():.4f}  편차={y[m].mean()-g:+.4f}')

np.save('dev/recovered_outing_idx.npy', outing_idx)
print(f'\n저장: dev/recovered_outing_idx.npy')
