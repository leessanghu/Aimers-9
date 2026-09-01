"""zone_intent를 성공측 분할축으로 쓸 수 있는지 클래스크기/순수성 확인. 학습 없음."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
n = len(y)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
order = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(n); lab[order] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
call = np.load('dev/recovered_call_axis.npy')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
zi = np.load('dev/recovered_zone_intent.npy')   # 성공행만 유효(그외 NaN), 존안=1/존밖=0 추정

valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
middle = valid & (mid > 0.5)
reverse = valid & (rev > 0.5) & (mid < 0.5)
nd = valid & (mid < 0.5) & (rev < 0.5)
succ = nd & (y == 1)
zi_ok = succ & np.isfinite(zi)

print(f'zone_intent 유효값 종류: {np.unique(zi[np.isfinite(zi)])}')
print(f'성공행 중 zone_intent 복원비율: {zi_ok.sum()}/{succ.sum()} = {zi_ok.mean()*100:.1f}% (분모=성공행)')
print(f'전체행 대비: {zi_ok.mean()*100:.2f}%\n')

cls = np.full(n, -1, dtype=np.int64)
cls[middle] = 0
cls[reverse] = 1
cls[nd & (y == 0)] = 2
cls[zi_ok & (zi < 0.5)] = 3   # 존밖요구 성공
cls[zi_ok & (zi >= 0.5)] = 4  # 존안요구 성공
names = ['middle', 'reverse', 'wild', 'succ_out존', 'succ_in존']
for c in range(5):
    m = cls == c
    print(f'  {c} {names[c]:<12} n={m.sum():>9,} ({m.mean()*100:5.2f}%)  성공률={y[m].mean()*100:6.2f}%')
print(f'  미분류: {(cls<0).sum():,} ({(cls<0).mean()*100:.2f}%)  (성공인데 zone_intent 결측인 행 포함)')
