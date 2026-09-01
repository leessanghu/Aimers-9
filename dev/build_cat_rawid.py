"""CatBoost + raw ID, xgb_rawid/lgbm_rawid와 완전히 동일한 설정(162피처+ID, binary logloss).
HGB(base헤드)/CatBoost/XGB/LGBM 4개 알고리즘 공정비교용."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807
X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
pid = raw_all['pitcher_id'].to_numpy()
bid = raw_all['batter_id'].to_numpy()
ptid = raw_all['pitcher_team_id'].to_numpy()
btid = raw_all['batter_team_id'].to_numpy()

CB = dict(iterations=3000, learning_rate=0.01, depth=7, l2_leaf_reg=5.0, verbose=0,
          loss_function='Logloss', early_stopping_rounds=100, random_seed=42,
          subsample=0.9, colsample_bylevel=0.6)


def build_fold(upto, vs):
    tr = season <= upto
    va = season == vs
    Xtr = X.loc[tr].copy().astype(np.float64)
    Xva = X.loc[va].copy().astype(np.float64)
    for col, arr in [('pitcher_id_raw', pid), ('batter_id_raw', bid),
                     ('pitcher_team_id_raw', ptid), ('batter_team_id_raw', btid)]:
        Xtr[col] = pd.Categorical(arr[tr]).codes.astype(np.float64)
        cats_tr = pd.Categorical(arr[tr]).categories
        Xva[col] = pd.Categorical(arr[va], categories=cats_tr).codes.astype(np.float64)
    return Xtr, Xva, y[tr], y[va], tr, va


def train_eval(upto, vs, tag):
    Xtr, Xva, ytr, yva, tr, va = build_fold(upto, vs)
    w = 0.5 ** ((upto - season[tr]) / 2.0)
    n_es = int(len(Xtr) * 0.92)
    m = CatBoostClassifier(**CB)
    m.fit(Xtr.iloc[:n_es], ytr[:n_es], sample_weight=w[:n_es],
          eval_set=(Xtr.iloc[n_es:], ytr[n_es:]))
    p = np.clip(m.predict_proba(Xva)[:, 1], 0, 1)
    log(f'[{tag}] 학습완료 best_iter={m.best_iteration_}')
    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yva) ** 2) / B)
    print(f'  단독 BSS = {sc(p):.2f}')
    return p


log('=== fold A (train<=2023 -> 2024) ===')
pA = train_eval(2023, 2024, 'A')
np.save('dev/cache_catrawid_A.npy', pA)

log('=== fold C (train<=2021 -> 2022) ===')
pC = train_eval(2021, 2022, 'C')
np.save('dev/cache_catrawid_C.npy', pC)
log('완료')
