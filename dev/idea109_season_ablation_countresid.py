"""season 의존도가 큰 헤드(countresid, rank2 imp=11.68)로 season 포함/제외 손해 재확인.
idea54_new_axes.py의 countresid 설정 그대로 재현(fold A, train<=2023->2024)."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season_arr = meta['season'].to_numpy()
count_state = X['count_state'].to_numpy()
unc = 0.249807

tr_m = season_arr <= 2023
va_m = season_arr == 2024
yv = y[va_m]
w = 0.5 ** ((2023 - season_arr) / 2.0)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / unc)

ctab = pd.DataFrame({'cs': count_state[tr_m], 'y': y[tr_m]}).groupby('cs')['y'].agg(['sum', 'count'])
K_C = 500.0
ctab['prior'] = (ctab['sum'] + K_C * y[tr_m].mean()) / (ctab['count'] + K_C)
cprior_all = pd.Series(count_state).map(ctab['prior']).fillna(y[tr_m].mean()).to_numpy(np.float64)
h_count_resid = y - cprior_all
Ymat = np.column_stack([y, h_count_resid])

n_es = int(tr_m.sum() * 0.92)
CAT = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50)

def train_eval(drop_season, seed=42):
    feats = [c for c in X.columns if not (drop_season and c == 'season')]
    m = CatBoostRegressor(**CAT, random_seed=seed)
    m.fit(X.loc[tr_m, feats].iloc[:n_es], Ymat[tr_m][:n_es], sample_weight=w[tr_m][:n_es],
          eval_set=(X.loc[tr_m, feats].iloc[n_es:], Ymat[tr_m][n_es:]))
    p = np.clip(m.predict(X.loc[va_m, feats]), 0.0, 1.0)[:, 0]
    return sc(p), m.best_iteration_

log('season 포함 학습...')
s_with, it_with = train_eval(False)
log(f'season 포함  countresid단독={s_with:.2f}  best_iter={it_with}')

log('season 제외 학습...')
s_without, it_without = train_eval(True)
log(f'season 제외  countresid단독={s_without:.2f}  best_iter={it_without}')

print(f'\n델타(제외 - 포함) = {s_without - s_with:+.2f}')
log('완료')
