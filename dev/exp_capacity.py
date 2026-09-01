"""모델 용량(depth/leaves) 가설 검증: 피처는 그대로, 트리 용량만 키우면 fold A가 오르는가?
현재 프로덕션 CatBoost는 depth=6~8, l2=5~10. 같은 피처(162개)로 depth/leaves를 키운
버전을 early stopping으로 공정 비교. 오르면 '모델이 세밀한 상호작용을 놓치고 있다'는
가설을 지지, 안 오르면 '깊이는 문제가 아니다'."""
import numpy as np, pandas as pd, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from catboost import CatBoostClassifier
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
row_num = meta['row_num'].to_numpy()
tr = season <= 2023; va = season == 2024
Xt = X.loc[tr].reset_index(drop=True); yt = y[tr]
Xv, yv = X.loc[va], y[va]
unc = 0.249807
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / unc)
rn = row_num[tr]
order = np.argsort(rn); n_es = int(len(Xt) * 0.92)
ti, ei = order[:n_es], order[n_es:]
w = 0.5 ** ((2023 - season[tr].astype(float)) / 2.0)   # 프로덕션 recency 유지

CONFIGS = [
    ('depth=6,l2=5 (현재프로덕션)', dict(depth=6, l2_leaf_reg=5.0)),
    ('depth=8,l2=5',               dict(depth=8, l2_leaf_reg=5.0)),
    ('depth=10,l2=5',              dict(depth=10, l2_leaf_reg=5.0)),
    ('depth=10,l2=10(정규화도같이)', dict(depth=10, l2_leaf_reg=10.0)),
    ('depth=12,l2=10',             dict(depth=12, l2_leaf_reg=10.0)),
]
for name, extra in CONFIGS:
    params = dict(iterations=3000, learning_rate=0.03, loss_function='Logloss',
                 random_seed=42, verbose=False, min_data_in_leaf=200, early_stopping_rounds=50)
    params.update(extra)
    m = CatBoostClassifier(**params)
    m.fit(Xt.iloc[ti], yt[ti], sample_weight=w[ti], eval_set=(Xt.iloc[ei], yt[ei]))
    p = m.predict_proba(Xv)[:, 1]
    log(f'{name:32s} best_iter={m.get_best_iteration():>4}  BSS={sc(p):8.1f}')
