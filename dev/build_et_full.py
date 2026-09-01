"""ExtraTrees 프로덕션: 전체데이터(2019-2024). fold A z=1.8(경계). 방향 양수(로컬).
피클 크기 주의: min_samples_leaf=200, 300트리 - 저장 후 크기 확인."""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import ExtraTreesClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
Xm = X_df[FEAT].fillna(0).to_numpy()

w = 0.5 ** ((2024.0 - season) / 2.0)
log('학습 시작...')
et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=200, max_features=0.4,
                          n_jobs=-1, random_state=42)
et.fit(Xm, y, sample_weight=w)
log('학습완료')
joblib.dump(dict(model=et, feat_order=FEAT), 'dev/et_production.pkl', compress=3)
sz = os.path.getsize('dev/et_production.pkl') / 1e6
log(f'저장 완료: dev/et_production.pkl ({sz:.1f}MB)')
