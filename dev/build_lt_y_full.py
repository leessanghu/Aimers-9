"""lt_y 프로덕션: linear_tree LGBM, binary y, 전체데이터(2019-2024).

fold A 스크리닝(2026-08-31): 직교화후 rho=-0.00636, z=4.1 통과(오늘 최강).
근거: HGB/CatBoost/XGB/기본LGBM 전부 계단함수(piecewise-constant)인데
     linear_tree는 리프내 선형회귀 -> 함수공간 자체가 다름. mc6/xu와 직교화 후에도
     신호 생존 = 계단모델들이 공통으로 놓치는 매끄러운 기울기 성분.
방향: 음수(빼기). 프로브 가중치는 별도 빌드에서 -0.03.

하이퍼파라미터는 fold 실험(build_lgbm_lineartree.py)과 동일 - 바꾸면 검증 무효.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import lightgbm
from lightgbm import LGBMClassifier, early_stopping

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

print(f'lightgbm 버전 = {lightgbm.__version__} (requirements.txt와 일치해야 함: 4.7.0)')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)

# fold 실험과 완전히 동일한 하이퍼파라미터 (LT_COMMON)
LT = dict(linear_tree=True, n_estimators=500, learning_rate=0.05,
          num_leaves=31, min_child_samples=200, subsample=0.9, subsample_freq=1,
          colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=5.0, random_state=42,
          n_jobs=-1, verbose=-1, objective='binary')

w = 0.5 ** ((2024.0 - season) / 2.0)
n_es = int(len(X) * 0.92)

log('전체데이터 학습 시작...')
m = LGBMClassifier(**LT)
m.fit(X.iloc[:n_es], y[:n_es], sample_weight=w[:n_es],
      eval_set=[(X.iloc[n_es:], y[n_es:])],
      callbacks=[early_stopping(50, verbose=False)])
log(f'학습완료 best_iter={m.best_iteration_}')

# 저장 전 셀프테스트: joblib 왕복 후 예측 일치 확인 (linear_tree 직렬화 검증)
p_before = m.predict_proba(X.iloc[:1000])[:, 1]
joblib.dump(dict(model=m, feat_order=FEAT), 'dev/lty_production.pkl')
rt = joblib.load('dev/lty_production.pkl')
p_after = rt['model'].predict_proba(X.iloc[:1000][rt['feat_order']])[:, 1]
assert np.allclose(p_before, p_after, atol=1e-12), 'joblib 왕복 후 예측 불일치!'
log(f'직렬화 왕복 검증 통과 (최대차이 {np.abs(p_before-p_after).max():.2e})')
log('저장 완료: dev/lty_production.pkl')
