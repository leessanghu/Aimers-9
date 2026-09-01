"""season 피처가 2025(모델이 한번도 못 본 값)로 갈 때 일반화를 해치는지,
mc5(season 중요도가 가장 크게 나온 헤드)로 fold A(train<=2023->2024) 정직 테스트.
season 포함 vs 제외 비교."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
cls5 = np.load('dev/cls5_labels.npy')
pt = np.load('dev/pitchtype_labels.npy')
unc = 0.249807

v = (cls5 >= 0) & (pt >= 0)
cls = np.full(len(cls5), -1, dtype=np.int64)
nd = v & (cls5 >= 2); cls[nd] = (cls5[nd] - 2) * 3 + pt[nd]
cls[v & (cls5 == 0)] = 9; cls[v & (cls5 == 1)] = 10

tr = season <= 2023; va = season == 2024
fit = tr & (cls >= 0)
w = 0.5 ** ((2023 - season) / 2.0)
fi = np.where(fit)[0]; n_es = int(len(fi) * 0.92)
ti, ei = fi[:n_es], fi[n_es:]

y = meta['control_success'].to_numpy(np.float64)
yv = y[va]
v88 = joblib_mc5_succ = __import__('joblib').load('submit/model/model_artifacts_v88.pkl')
mc5_succ = np.asarray(v88['mc5_succ'], dtype=np.float64)
sc = lambda p_: 1e5 * (1 - np.mean((np.clip(p_, 0, 1) - yv) ** 2) / unc)

def train_eval(drop_season):
    feats = [c for c in X.columns if not (drop_season and c == 'season')]
    m = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
                            verbose=False, random_seed=42, loss_function='MultiClass',
                            classes_count=11, early_stopping_rounds=40)
    m.fit(X.iloc[ti][feats], cls[ti], sample_weight=w[ti], eval_set=(X.iloc[ei][feats], cls[ei]))
    P = m.predict_proba(X.loc[va, feats])
    p_mc5 = np.clip(P @ mc5_succ, 0, 1)
    return sc(p_mc5), m.best_iteration_

log('season 포함 학습...')
s_with, it_with = train_eval(False)
log(f'season 포함  mc5단독={s_with:.2f}  best_iter={it_with}')

log('season 제외 학습...')
s_without, it_without = train_eval(True)
log(f'season 제외  mc5단독={s_without:.2f}  best_iter={it_without}')

print(f'\n델타(제외 - 포함) = {s_without - s_with:+.2f}')
log('완료')
