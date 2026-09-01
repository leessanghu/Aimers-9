"""raw 포렌식 3차 — mc6 성공의 논리를 확장.

mc6가 통한 이유(가설): mc5가 'nd&ball'처럼 성공/실패를 뭉쳐놓은 걸 풀어줬다.
=> 같은 논리로 "아직 뭉쳐 있는 축"을 찾는다.

이번엔 '시퀀스/맥락' 축을 본다. 지금까지 우리가 쓴 라벨은 전부 '이 투구 하나'의 결과였다.
raw엔 투구 순서(row_num)와 as-of 카운터가 있어서 '직전 투구가 뭐였나'를 복원할 수 있다.
직전 투구의 결과는 test에서도 알 수 있나? -> 아니다(다른 행 참조 금지).
하지만 학습 보조타겟으로는 쓸 수 있다. 더 중요한 건:
  '같은 타석/같은 이닝 안에서 이 투구가 몇 번째인가'는 as-of 카운터가 아니라
  balls/strikes/outs/runner 상태에서 '유도'할 수 있고 이건 test에서도 계산 가능하다.

확인할 것:
 (1) 타석 내 투구순번(balls+strikes)별 성공률 - 이미 count_state로 쓰고 있나?
 (2) 직전 투구 결과 복원 가능성 + 그 조건부 성공률 (보조타겟 후보)
 (3) 연속 실패(2연속/3연속) 상태의 성공률 - '흔들림' 신호
 (4) 이닝 내 누적 투구수 (피로) - as-of와 다른 축
 (5) 같은 타자 상대 반복 (2번째/3번째 대결)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
y = df['control_success'].to_numpy(np.float64)
pid = df['pitcher_id'].to_numpy()
bid = df['batter_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
balls = df['balls_before'].to_numpy()
strikes = df['strikes_before'].to_numpy()
inning = df['inning'].to_numpy()
outs = df['outs_before'].to_numpy()
g = y.mean()

order = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])

print(f'전역 성공률 = {g:.4f}\n')

print('=' * 78)
print('[1] 타석 내 투구순번 (balls+strikes) 별 성공률')
print('=' * 78)
pa_pitch = balls + strikes
for k in range(6):
    m = pa_pitch == k
    if m.sum() < 1000:
        continue
    print(f'  {k}번째 투구  n={m.sum():>9,} ({m.mean()*100:5.2f}%)  성공률={y[m].mean():.4f}  '
          f'편차={y[m].mean()-g:+.4f}')

print('\n' + '=' * 78)
print('[2] 직전 투구 결과 복원 (같은 투수 연속행) -> 조건부 성공률')
print('=' * 78)
# 직전 투구의 y (같은 투수, n이 정확히 1 증가한 경우)
prev_y = np.full(n, np.nan)
o = order
dn = np.full(n, np.nan)
dn_ord = np.empty(len(o)); dn_ord[1:] = n_[o][1:] - n_[o][:-1]; dn_ord[0] = np.nan
same_prev = np.zeros(len(o), dtype=bool)
same_prev[1:] = (pid[o][1:] == pid[o][:-1])
prev_y_ord = np.full(len(o), np.nan)
prev_y_ord[1:] = y[o][:-1]
valid_prev = same_prev & (dn_ord == 1)
tmp = np.where(valid_prev, prev_y_ord, np.nan)
prev_y[o] = tmp
ok_prev = np.isfinite(prev_y)
print(f'  직전투구 복원 가능: {ok_prev.sum():,} ({ok_prev.mean()*100:.1f}%)')
for v, nm in [(1.0, '직전 성공'), (0.0, '직전 실패')]:
    m = ok_prev & (prev_y == v)
    print(f'  {nm}  n={m.sum():>9,}  현재 성공률={y[m].mean():.4f}  편차={y[m].mean()-g:+.4f}')
d_streak = y[ok_prev & (prev_y == 1)].mean() - y[ok_prev & (prev_y == 0)].mean()
print(f'  => 직전성공/실패 성공률 격차 = {d_streak:+.4f} ({d_streak*100:.2f}%p)')

print('\n' + '=' * 78)
print('[3] 연속 실패 길이별 성공률 (흔들림 신호)')
print('=' * 78)
# 같은 투수 내 연속 실패 카운트(직전까지)
streak = np.zeros(len(o))
cur = 0.0
for i in range(len(o)):
    if i == 0 or not (same_prev[i] and dn_ord[i] == 1):
        cur = 0.0
    streak[i] = cur
    cur = 0.0 if y[o][i] == 1 else cur + 1
fail_streak = np.empty(n); fail_streak[o] = streak
for k in range(5):
    m = fail_streak == k
    if m.sum() < 1000:
        continue
    print(f'  직전 {k}연속실패  n={m.sum():>9,}  성공률={y[m].mean():.4f}  편차={y[m].mean()-g:+.4f}')
m = fail_streak >= 5
if m.sum() > 500:
    print(f'  직전 5+연속실패 n={m.sum():>9,}  성공률={y[m].mean():.4f}  편차={y[m].mean()-g:+.4f}')

print('\n' + '=' * 78)
print('[4] 같은 타자와 몇 번째 대결인가 (경기 내 반복)')
print('=' * 78)
sub = pd.DataFrame({'p': pid, 'b': bid, 'rn': df['row_num'].to_numpy(), 'y': y})
sub = sub.sort_values('rn')
# (투수,타자) 조합의 등장 순번 - 단, 경기구분자가 없어 전체 커리어 기준
sub['nth'] = sub.groupby(['p', 'b']).cumcount()
nth = sub['nth'].reindex(range(n)).to_numpy()
for k in [0, 1, 2, 3]:
    m = nth == k
    if m.sum() < 2000:
        continue
    print(f'  {k+1}번째 대결  n={m.sum():>9,}  성공률={y[m].mean():.4f}  편차={y[m].mean()-g:+.4f}')
m = nth >= 10
print(f'  11번째+      n={m.sum():>9,}  성공률={y[m].mean():.4f}  편차={y[m].mean()-g:+.4f}')

print('\n' + '=' * 78)
print('[5] 이닝 x 아웃 상태 (이미 쓰는 축이지만 조합으로)')
print('=' * 78)
for inn in [1, 3, 5, 7, 9]:
    m = inning == inn
    if m.sum() < 2000:
        continue
    print(f'  {inn}회  n={m.sum():>9,}  성공률={y[m].mean():.4f}  편차={y[m].mean()-g:+.4f}')

print('\n' + '=' * 78)
print('[요약] 편차가 큰 축 = 새 하위분할 후보')
print('=' * 78)
print('  직전투구 결과는 test에서 계산 불가(다른 행 참조) -> 보조타겟으로만 사용 가능')
print('  타석내 투구순번/이닝/아웃은 test에서 계산 가능 -> 피처 또는 분할축 가능')
