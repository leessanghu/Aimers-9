"""로컬 test.csv에 대해 v95로 실제 예측 + 컬럼/분포 점검.
로컬 test.csv는 5행짜리 샘플이라 분포 비교는 제한적이지만
(a) 컬럼 구성이 train과 동일한지 (b) 값 범위가 정상인지 (c) 결측 패턴을 본다."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

test = pd.read_csv('data/test.csv', encoding='utf-8-sig')
train = pd.read_csv('data/train.csv', encoding='utf-8-sig', nrows=200000)

print(f'=== test.csv: {test.shape[0]}행 x {test.shape[1]}컬럼 ===')
tr_cols = set(train.columns) - {'control_success'}
te_cols = set(test.columns)
print(f'  train(타깃제외)에만 있음: {sorted(tr_cols - te_cols)}')
print(f'  test에만 있음          : {sorted(te_cols - tr_cols)}')
print(f'  공통                   : {len(tr_cols & te_cols)}개')

print(f'\n=== test.csv 전체 내용 ===')
pd.set_option('display.width', 250)
pd.set_option('display.max_columns', 100)
print(test.to_string())

print(f'\n=== 결측 현황 ===')
na = test.isna().sum()
na = na[na > 0]
print(na.to_string() if len(na) else '  결측 없음')

print(f'\n=== season 값 ===')
print(f"  test season: {sorted(test['season'].unique())}")
print(f"  train season 범위: {train['season'].min()}~{train['season'].max()}")

print(f'\n=== 주요 수치 컬럼: test 값 vs train(2024) 범위 ===')
tr24 = pd.read_csv('data/train.csv', encoding='utf-8-sig')
tr24 = tr24[tr24['season'] == 2024]
CHK = ['asof_pitcher_n', 'asof_pitcher_success_rate', 'asof_pitcher_middle_rate',
       'asof_batter_n', 'asof_pitcher_pitchmix_n', 'li', 'inning']
for c in CHK:
    if c not in test.columns:
        continue
    tv = test[c].to_numpy(np.float64)
    rv = tr24[c].to_numpy(np.float64)
    print(f'  {c:32s} test={np.array2string(tv, precision=3)}   '
          f'train2024 [{np.nanmin(rv):.3f}, {np.nanmax(rv):.3f}] 중앙값={np.nanmedian(rv):.3f}')

print(f'\n=== 투수/타자 ID가 train에 있는가 (콜드스타트 점검) ===')
tr_pid = set(tr24['pitcher_id'].unique())
tr_bid = set(tr24['batter_id'].unique())
for _, r in test.iterrows():
    print(f"  row={r['row_id']}  pitcher_id={r['pitcher_id']}(train2024에 있음:{r['pitcher_id'] in tr_pid})  "
          f"batter_id={r['batter_id']}(있음:{r['batter_id'] in tr_bid})")
