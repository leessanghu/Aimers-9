"""middle/reverse가 판정축(ball/strike/inplay)과 교차하는지, 순수분할이 되는지 확인.
mc6와 상관도 미리 대략 점검(전체 성공률 패턴으로 근사)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
y = df['control_success'].to_numpy(np.float64)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
o = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[o[:-1]] = (pid[o][1:] == pid[o][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[o]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[o]] = np.nan
    lab = np.empty(n); lab[o] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
call = np.load('dev/recovered_call_axis.npy')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)

middle = valid & (mid > 0.5)
reverse = valid & (rev > 0.5) & (mid < 0.5)

print(f'전체 middle: {middle.sum():,}  전체 reverse: {reverse.sum():,}\n')

print('=== middle x 판정축 교차 ===')
for nm, m in [('middle&ball', middle & (ball > 0.5)),
              ('middle&strike', middle & (strike > 0.5)),
              ('middle&inplay', middle & (inplay > 0.5))]:
    print(f'  {nm:<16} n={m.sum():>9,}  middle중 비중={m.sum()/middle.sum()*100:5.2f}%  '
          f'성공률={y[m].mean()*100:.2f}%')

print('\n=== reverse x 판정축 교차 ===')
for nm, m in [('reverse&ball', reverse & (ball > 0.5)),
              ('reverse&strike', reverse & (strike > 0.5)),
              ('reverse&inplay', reverse & (inplay > 0.5))]:
    print(f'  {nm:<16} n={m.sum():>9,}  reverse중 비중={m.sum()/reverse.sum()*100:5.2f}%  '
          f'성공률={y[m].mean()*100:.2f}%')

print('\n=== 순수성 검증: 이 세분류에서도 성공률이 정확히 0%인가? ===')
allpure = True
for nm, m in [('middle&ball', middle & (ball > 0.5)), ('middle&strike', middle & (strike > 0.5)),
              ('middle&inplay', middle & (inplay > 0.5)), ('reverse&ball', reverse & (ball > 0.5)),
              ('reverse&strike', reverse & (strike > 0.5)), ('reverse&inplay', reverse & (inplay > 0.5))]:
    pure = y[m].mean() == 0.0
    allpure = allpure and pure
    if not pure:
        print(f'  [!] {nm}: 순수아님, 성공률={y[m].mean()*100:.4f}%')
print(f'  전부 순수(0%)? {allpure}')

print('\n=== 새로운 10클래스 구조 제안 ===')
labels = ['middle&ball', 'middle&strk', 'middle&play', 'reverse&ball', 'reverse&strk',
          'reverse&play', 'succ_ball', 'succ_strk', 'succ_play']
comps = [middle & (ball > 0.5), middle & (strike > 0.5), middle & (inplay > 0.5),
         reverse & (ball > 0.5), reverse & (strike > 0.5), reverse & (inplay > 0.5)]
wild = valid & (mid < 0.5) & (rev < 0.5) & (y == 0)
comps.append(wild)
labels.append('wild')
succ = valid & (mid < 0.5) & (rev < 0.5) & (y == 1)
comps += [succ & (ball > 0.5), succ & (strike > 0.5), succ & (inplay > 0.5)]
labels += ['succ_ball', 'succ_strk', 'succ_play']
tot = 0
for nm, m in zip(labels, comps):
    print(f'  {nm:<14} n={m.sum():>9,} ({m.sum()/n*100:5.2f}%)  성공률={y[m].mean()*100:6.2f}%')
    tot += m.sum()
print(f'  합계 커버리지 = {tot/n*100:.2f}%  (10클래스, wild는 미분할 유지)')
