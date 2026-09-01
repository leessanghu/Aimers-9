"""CatBoost/LGBM/HGB/RandomForest를 동일조건(전체162피처, train<=2023->2024,
비슷한 깊이/정규화)으로 학습해서 확인된 약점 2곳(완전신인 n=0, 0-2카운트)에서
어느 모델이 신호를 잘 잡는지 비교."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807
X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
count_state = (raw_all['balls_before'] * 4 + raw_all['strikes_before']).to_numpy()
n_pitcher = np.nan_to_num(raw_all['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)

tr = season <= 2023
va = season == 2024
yv = y[va]
w = 0.5 ** ((2023 - season) / 2.0)
FEATS = list(X.columns)

ti_all = np.where(tr)[0]
n_es = int(len(ti_all) * 0.92)
ti, ei = ti_all[:n_es], ti_all[n_es:]

sc_arr = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)
preds = {}

log('CatBoost 학습...')
m_cb = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                           loss_function='Logloss', verbose=False, random_seed=42,
                           min_data_in_leaf=200, early_stopping_rounds=50)
m_cb.fit(X.iloc[ti], y[ti], sample_weight=w[ti], eval_set=(X.iloc[ei], y[ei]))
preds['CatBoost'] = np.clip(m_cb.predict_proba(X.loc[va])[:, 1], 0, 1)
log(f'  CatBoost 완료 best_iter={m_cb.get_best_iteration()}')

log('LightGBM 학습...')
params = dict(objective='binary', metric='binary_logloss', learning_rate=0.03,
              num_leaves=63, max_depth=6, min_data_in_leaf=200, lambda_l2=5.0,
              verbose=-1, seed=42)
dtr = lgb.Dataset(X.iloc[ti], y[ti], weight=w[ti])
dva = lgb.Dataset(X.iloc[ei], y[ei], weight=w[ei], reference=dtr)
m_lgb = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                   callbacks=[lgb.early_stopping(50, verbose=False)])
preds['LightGBM'] = np.clip(m_lgb.predict(X.loc[va]), 0, 1)
log(f'  LightGBM 완료 best_iter={m_lgb.best_iteration}')

log('HGB 학습...')
m_hgb = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=63, max_iter=1000,
                                        learning_rate=0.03, l2_regularization=5.0,
                                        min_samples_leaf=200, early_stopping=True,
                                        validation_fraction=0.08, random_state=42)
m_hgb.fit(X.iloc[ti_all], y[ti_all], sample_weight=w[ti_all])
preds['HGB'] = np.clip(m_hgb.predict_proba(X.loc[va])[:, 1], 0, 1)
log(f'  HGB 완료 n_iter={m_hgb.n_iter_}')

log('RandomForest 학습 (부스팅 아님, 배깅)...')
m_rf = RandomForestClassifier(n_estimators=500, max_depth=10, min_samples_leaf=200,
                               n_jobs=-1, random_state=42)
m_rf.fit(X.iloc[ti_all], y[ti_all], sample_weight=w[ti_all])
preds['RandomForest'] = np.clip(m_rf.predict_proba(X.loc[va])[:, 1], 0, 1)
log('  RandomForest 완료')

# 우리 블렌드도 비교군에 추가
preds['우리v88_final'] = np.load('dev/cache_v88_final_2024.npy')

is_rookie = n_pitcher[va] < 0.5
is02 = count_state[va] == 2
allm = np.ones(len(yv), bool)

groups = [('전체', allm), ('완전신인(n=0)', is_rookie), ('0-2카운트', is02)]

print(f'\n{"모델":16s}', end='')
for gname, _ in groups:
    print(f'{gname:>18s}', end='')
print()
for name, p in preds.items():
    print(f'{name:16s}', end='')
    for gname, m in groups:
        yy = yv[m]; pp = p[m]
        r = yy.mean(); var_own = max(r * (1 - r), 1e-6)
        bs = np.mean((pp - yy) ** 2)
        bss_own = 1e5 * (1 - bs / var_own)
        bias = pp.mean() - r
        print(f'  {bss_own:7.1f}(편차{bias:+.4f})', end='')
    print()

log('완료')
