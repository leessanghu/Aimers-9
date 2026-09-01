import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np

meta = pd.read_parquet('dev/featcache_meta.parquet')
print('featcache_meta 컬럼:', list(meta.columns))

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')

# season 일치는 이미 확인됨. row_id/pitcher_id/control_success까지 완전 일치하는지 확인.
checks = {}
if 'row_id' in meta.columns:
    checks['row_id'] = bool((meta['row_id'].to_numpy() == df['row_id'].to_numpy()).all())
if 'pitcher_id' in meta.columns:
    checks['pitcher_id'] = bool((meta['pitcher_id'].to_numpy() == df['pitcher_id'].to_numpy()).all())
checks['control_success'] = bool((meta['control_success'].to_numpy() == df['control_success'].to_numpy()).all())
checks['season'] = bool((meta['season'].to_numpy() == df['season'].to_numpy()).all())
checks['len'] = len(meta) == len(df)

for k, v in checks.items():
    print(f'  {k}: {"일치" if v else "!!불일치!!"}')

if not all(checks.values()):
    print('\n!!! 정렬 불일치 발견 - 아래에서 첫 어긋난 지점 확인 !!!')
    if 'row_id' in meta.columns:
        mism = np.flatnonzero(meta['row_id'].to_numpy() != df['row_id'].to_numpy())
        print(f'  row_id 어긋난 행 수: {len(mism)}  (예시 idx: {mism[:5]})')
else:
    print('\n=> featcache_X.parquet 행순서 = data/train.csv 행순서. 완전 일치. 정렬버그 없음.')
