"""codex v20 패키지를 fold A(2024) 행에 돌리기 위한 입력 준비.
주의: 이 모델은 correction_beta를 2024로 적합했다고 metadata에 명시돼 있음.
따라서 2024 예측은 '오염된(낙관적) 상한'이다. 이 성질을 역이용한다:
  오염 = 잔차상관을 인위적으로 양수쪽으로 부풀림
  -> 오염된 상태에서조차 잔차상관이 0근처/음수면 => 확정 기각 가능(단방향 검정)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, os

test_real = pd.read_csv('data/test.csv', encoding='utf-8-sig', nrows=5)
train = pd.read_csv('data/train.csv', encoding='utf-8-sig')

print('test.csv 컬럼:', len(test_real.columns))
print('train.csv 컬럼:', len(train.columns))
extra = [c for c in train.columns if c not in test_real.columns]
miss = [c for c in test_real.columns if c not in train.columns]
print('train에만 있는 컬럼:', extra)
print('test에만 있는 컬럼:', miss)

fold = train[train['season'] == 2024].reset_index(drop=True)
out = fold[[c for c in test_real.columns]].copy()
dest = 'dev/codex_foldA_input'
os.makedirs(dest, exist_ok=True)
out.to_csv(f'{dest}/test.csv', index=False, encoding='utf-8')
pd.DataFrame({'row_id': out['row_id'], 'control_success': 0.5}).to_csv(
    f'{dest}/sample_submission.csv', index=False, encoding='utf-8')
np.save('dev/codex_foldA_y.npy', fold['control_success'].to_numpy(np.float64))
print(f'\n저장완료: {dest}/test.csv  rows={len(out)}')
