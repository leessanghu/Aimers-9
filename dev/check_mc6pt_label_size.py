"""mc6pt(성공측을 구종축으로 재분할) 클래스크기 사전확인. 학습 없이 라벨만."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

X = pd.read_parquet('dev/featcache_X.parquet')
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

# 구종축 복원 확인 (raw-recoverable-labels 메모리 기준: fastball/breaking/offspeed rate diff)
fast_lab = diff_label('asof_pitcher_fastball_rate_smooth') if 'asof_pitcher_fastball_rate_smooth' in df.columns else None
print('구종 관련 컬럼 탐색:')
for c in df.columns:
    if 'fastball' in c or 'breaking' in c or 'offspeed' in c:
        print(f'  {c}')
