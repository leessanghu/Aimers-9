"""1) rsm(피처 다양성) 스윕 — 강한 피처 의존을 강제로 낮췄을 때 fold A 변화
2) pitcher_id/batter_id를 CatBoost 네이티브 범주형으로 직접 추가했을 때 변화
   (지금까지 원본 ID를 피처로 준 적이 없음 - count/smoothed rate로만 우회)
fold A: train<=2023 -> val 2024, early stopping 공정비교."""
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
w = 0.5 ** ((2023 - season[tr].astype(float)) / 2.0)

def fit_eval2(Xt_, Xv_, cat_idx, tag, extra=None):
    params = dict(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                 loss_function='Logloss', random_seed=42, verbose=False,
                 min_data_in_leaf=200, early_stopping_rounds=50)
    if extra:
        params.update(extra)
    m = CatBoostClassifier(**params)
    m.fit(Xt_.iloc[ti], yt[ti], sample_weight=w[ti], eval_set=(Xt_.iloc[ei], yt[ei]), cat_features=cat_idx)
    p = m.predict_proba(Xv_)[:, 1]
    log(f'{tag:36s} best_iter={m.get_best_iteration():>4}  BSS={sc(p):8.1f}')
    return p

print('=== 1) rsm(피처 다양성) 스윕, depth=6 고정 ===')
for rsm in [1.0, 0.8, 0.6, 0.4, 0.2]:
    fit_eval2(Xt, Xv, [], f'rsm={rsm}', extra=dict(rsm=rsm))

print()
print('=== 2) pitcher_id/batter_id를 원본 그대로 범주형 추가 ===')
raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['row_id', 'pitcher_id', 'batter_id'])
raw['row_num'] = raw['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
raw = raw.sort_values('row_num').reset_index(drop=True)
assert (raw['row_num'].to_numpy() == meta['row_num'].to_numpy()).all()

Xt_id = Xt.copy(); Xv_id = Xv.copy()
Xt_id['raw_pitcher_id'] = raw.loc[tr, 'pitcher_id'].astype(str).reset_index(drop=True).to_numpy()
Xt_id['raw_batter_id'] = raw.loc[tr, 'batter_id'].astype(str).reset_index(drop=True).to_numpy()
Xv_id['raw_pitcher_id'] = raw.loc[va, 'pitcher_id'].astype(str).to_numpy()
Xv_id['raw_batter_id'] = raw.loc[va, 'batter_id'].astype(str).to_numpy()
cat_cols = ['raw_pitcher_id', 'raw_batter_id']
cat_idx = [Xt_id.columns.get_loc(c) for c in cat_cols]
n_new_pitcher = len(set(Xv_id['raw_pitcher_id']) - set(Xt_id['raw_pitcher_id']))
n_new_batter = len(set(Xv_id['raw_batter_id']) - set(Xt_id['raw_batter_id']))
log(f'val에서 train에 없는 새 pitcher_id: {n_new_pitcher}, batter_id: {n_new_batter}')
fit_eval2(Xt_id, Xv_id, cat_idx, 'ID 원본 범주형 추가(+2피처)')

print()
print('=== 3) ID 범주형 + rsm 낮춤 병행 ===')
fit_eval2(Xt_id, Xv_id, cat_idx, 'ID추가 + rsm=0.6', extra=dict(rsm=0.6))
