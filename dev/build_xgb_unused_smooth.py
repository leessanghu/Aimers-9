"""안 쓰이는 피처(tm_matched/tm_lown_flag/pitcher_hand/form_missing) + season/game_type
조건화 + shrinkage 스무딩 성공률을 작은 XGBClassifier로 학습. 피처 적어서 빠름.
번들프로브 재료용 - 단독판정 아님, honest fold A/C만 빠르게 확인.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import xgboost as xgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
NEED_RHO = 0.01740

X_full = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)

RAW = ['tm_matched', 'tm_lown_flag', 'pitcher_hand', 'form_missing', 'cat_top_bottom',
       'season', 'cat_game_type']
Xr = X_full[RAW].astype(np.float64).copy()

# shrinkage 스무딩: (season, tm_matched) 그룹별 성공률, train-only, K=50
K_SHR = 50.0


def add_smooth(Xdf, tr_mask, key_cols):
    g_all = float(y[tr_mask].mean())
    key = pd.Series(list(zip(*[X_full[c].to_numpy()[tr_mask].astype(int) for c in key_cols])))
    ytr = y[tr_mask]
    stat = pd.DataFrame({'k': key, 'y': ytr}).groupby('k')['y'].agg(['sum', 'count'])
    smap = ((stat['sum'] + K_SHR * g_all) / (stat['count'] + K_SHR)).to_dict()
    key_all = pd.Series(list(zip(*[X_full[c].to_numpy().astype(int) for c in key_cols])))
    return key_all.map(smap).fillna(g_all).to_numpy(np.float64)


PARAMS = dict(n_estimators=1500, learning_rate=0.02, max_depth=5, min_child_weight=20,
              subsample=0.9, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
              max_bin=256, tree_method='hist', random_state=42, n_jobs=-1,
              early_stopping_rounds=80, objective='binary:logistic', eval_metric='logloss')

results = {}
for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    tr = season <= upto
    va = season == vs
    yv = y[va]
    w = 0.5 ** ((upto - season[tr]) / 2.0)

    Xd = Xr.copy()
    Xd['smooth_season_tmm'] = add_smooth(Xd, tr, ['season', 'tm_matched'])
    Xd['smooth_gtype_lown'] = add_smooth(Xd, tr, ['cat_game_type', 'tm_lown_flag'])

    Xtr, Xva = Xd.loc[tr], Xd.loc[va]
    n_es = int(tr.sum() * 0.92)
    ts = time.time()
    m = xgb.XGBClassifier(**PARAMS)
    m.fit(Xtr.iloc[:n_es], y[tr][:n_es], sample_weight=w[:n_es],
          eval_set=[(Xtr.iloc[n_es:], y[tr][n_es:])], verbose=False)
    p = np.clip(m.predict_proba(Xva)[:, 1], 0, 1)
    np.save(f'dev/cache_xgbunused_{tag}.npy', p)
    log(f'[{tag}] 학습완료 best_iter={m.best_iteration} ({time.time()-ts:.0f}s)')

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
    print(f'  단독 BSS = {sc(p):.2f}')
    results[tag] = p

log('완료 (번들프로브에서 v117기준 rho는 별도 스크립트로 계산)')
