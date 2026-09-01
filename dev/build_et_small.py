"""압축판 ExtraTrees: 트리 100, 리프 500 - 아티팩트 크기 축소용."""
import sys, time, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import ExtraTreesClassifier

t0 = time.time()
X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
Xm = X_df[FEAT].fillna(0).to_numpy()
w = 0.5 ** ((2024.0 - season) / 2.0)
et = ExtraTreesClassifier(n_estimators=100, min_samples_leaf=500, max_features=0.4,
                          n_jobs=-1, random_state=42)
et.fit(Xm, y, sample_weight=w)
joblib.dump(dict(model=et, feat_order=FEAT), 'dev/et_production_small.pkl', compress=3)
sz = os.path.getsize('dev/et_production_small.pkl') / 1e6
print(f'완료 {time.time()-t0:.0f}s, 크기 {sz:.1f}MB')

# 대형판과의 예측상관(fold A 검증행으로 축 방향 유지 확인)
big = joblib.load('dev/et_production.pkl')['model']
va = season == 2024
rows = np.where(va)[0][:100000]
p_small = et.predict_proba(Xm[rows])[:, 1]
p_big = big.predict_proba(Xm[rows])[:, 1]
print(f'대형판과 예측상관 = {np.corrcoef(p_small, p_big)[0, 1]:.4f}')
