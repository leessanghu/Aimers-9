"""v97 = v95 + mc5 시드배깅(순수 분산축소, 새 신호 아님 - 리스크 낮음).
production 전체데이터(2019-2024)로 mc5(11-class)를 seed 42/7/123 세 번 학습해
script.py에서 predict_proba 평균내도록 mc5_models 리스트로 저장.
v88의 mc5_model과 동일 config(depth=6, iterations=1000, lr=0.05, l2=5.0)."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(np.float64)
cls5 = np.load('dev/cls5_labels.npy')
pt = np.load('dev/pitchtype_labels.npy')

v = (cls5 >= 0) & (pt >= 0)
cls = np.full(len(cls5), -1, dtype=np.int64)
nd = v & (cls5 >= 2); cls[nd] = (cls5[nd] - 2) * 3 + pt[nd]
cls[v & (cls5 == 0)] = 9; cls[v & (cls5 == 1)] = 10

fit = cls >= 0
w = 0.5 ** ((2024 - season) / 2.0)  # production recency 기준(전체데이터 최신년도=2024)
fi = np.where(fit)[0]; n_es = int(len(fi) * 0.92)
ti, ei = fi[:n_es], fi[n_es:]
log(f'학습행 {fit.sum():,}')

SEEDS = [42, 7, 123]
models = []
for sd in SEEDS:
    m = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
                            verbose=False, random_seed=sd, loss_function='MultiClass',
                            classes_count=11, early_stopping_rounds=40)
    m.fit(X.iloc[ti], cls[ti], sample_weight=w[ti], eval_set=(X.iloc[ei], cls[ei]))
    log(f'seed{sd} 학습완료 best_iter={m.best_iteration_}')
    models.append(m)

_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")
def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for k, val in list(vars(obj).items()):
            if type(val).__name__ in _RNG_TYPES:
                setattr(obj, k, None)
            else:
                strip_rng(val, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for val in obj:
            strip_rng(val, seen, depth + 1)
for m in models:
    strip_rng(m)

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v97 = dict(v95)
v97['mc5_models'] = models
# mc5_model(단일)은 하위호환용으로 seed42 것만 남겨둠(사용은 안 되지만 get() 호출 안전성)
v97['mc5_model'] = models[0]
joblib.dump(v97, 'submit/model/model_artifacts_v97.pkl')
log('v97 저장 완료')
