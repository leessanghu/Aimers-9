import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib, pandas as pd

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
feats = v88['feature_order']
raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', nrows=2)
rawcols = [c for c in raw.columns if c not in ('row_id', 'control_success')]

print(f'=== 162개 파생피처 전체 ===')
for i in range(0, len(feats), 4):
    print('  ' + '  '.join(f'{f:36s}' for f in feats[i:i + 4]))

print(f'\n=== raw 컬럼 {len(rawcols)}개 중 이름이 그대로 피처에 있는 것 ===')
direct = [c for c in rawcols if c in feats]
missing = [c for c in rawcols if c not in feats]
print(f'  그대로 사용({len(direct)}): {direct}')
print(f'  이름 그대로는 없음({len(missing)}): {missing}')
