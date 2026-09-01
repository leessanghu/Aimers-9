"""codex v20을 fold C(2022)에도 돌려서 오염 정도를 진단.
correction_beta는 2024로 적합됐으므로 2022에 적용하면 '그 해에 맞춰진' 이득은 없다.
base 모델이 2019-2024 전체학습이면 2022도 in-sample이라 여전히 오염이지만,
2024만큼 심하진 않다. 두 fold의 오염배율을 비교하면 어느 쪽이 신뢰 가능한지 판별된다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import pandas as pd, numpy as np, os

test_real = pd.read_csv('data/test.csv', encoding='utf-8-sig', nrows=5)
train = pd.read_csv('data/train.csv', encoding='utf-8-sig')
fold = train[train['season'] == 2022].reset_index(drop=True)
out = fold[[c for c in test_real.columns]].copy()
dest = 'dev/codex_foldC_input'
os.makedirs(dest, exist_ok=True)
out.to_csv(f'{dest}/test.csv', index=False, encoding='utf-8')
pd.DataFrame({'row_id': out['row_id'], 'control_success': 0.5}).to_csv(
    f'{dest}/sample_submission.csv', index=False, encoding='utf-8')
print(f'저장완료 rows={len(out)}')
