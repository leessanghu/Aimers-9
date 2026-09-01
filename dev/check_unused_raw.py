import pandas as pd
import joblib

raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', nrows=5)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
feat_order = set(v95['feature_order'])

print(f'raw 컬럼 {len(raw.columns)}개')
for c in raw.columns:
    used = 'X' if any(c in f or f.startswith(c) for f in feat_order) else '  '
    print(f'  [{used}] {c}  (dtype={raw[c].dtype})')
