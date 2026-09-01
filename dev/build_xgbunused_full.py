"""xgb_unused 프로덕션 전체데이터(2019-2024) 학습. 오늘 실측 프로브용.
fold A/C 부호일치(둘 다 s*<0) -> 음수가중치로 v117에 추가.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import xgboost as xgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X_full = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)

RAW = ['tm_matched', 'tm_lown_flag', 'pitcher_hand', 'form_missing', 'cat_top_bottom',
       'season', 'cat_game_type']
Xr = X_full[RAW].astype(np.float64).copy()
K_SHR = 50.0


def add_smooth(Xdf, tr_mask, key_cols):
    g_all = float(y[tr_mask].mean())
    key = pd.Series(list(zip(*[X_full[c].to_numpy()[tr_mask].astype(int) for c in key_cols])))
    ytr = y[tr_mask]
    stat = pd.DataFrame({'k': key, 'y': ytr}).groupby('k')['y'].agg(['sum', 'count'])
    smap = ((stat['sum'] + K_SHR * g_all) / (stat['count'] + K_SHR)).to_dict()
    key_all = pd.Series(list(zip(*[X_full[c].to_numpy().astype(int) for c in key_cols])))
    return key_all.map(smap).fillna(g_all).to_numpy(np.float64)


tr_all = np.ones(len(y), dtype=bool)  # 전체데이터로 학습(프로덕션)
Xd = Xr.copy()
Xd['smooth_season_tmm'] = add_smooth(Xd, tr_all, ['season', 'tm_matched'])
Xd['smooth_gtype_lown'] = add_smooth(Xd, tr_all, ['cat_game_type', 'tm_lown_flag'])
FEAT_XU = list(Xd.columns)

w = 0.5 ** ((2024.0 - season) / 2.0)
n_es = int(len(Xd) * 0.92)

PARAMS = dict(n_estimators=1500, learning_rate=0.02, max_depth=5, min_child_weight=20,
              subsample=0.9, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
              max_bin=256, tree_method='hist', random_state=42, n_jobs=-1,
              early_stopping_rounds=80, objective='binary:logistic', eval_metric='logloss')

log('전체데이터 학습 시작...')
m = xgb.XGBClassifier(**PARAMS)
m.fit(Xd.iloc[:n_es], y[:n_es], sample_weight=w[:n_es],
      eval_set=[(Xd.iloc[n_es:], y[n_es:])], verbose=False)
log(f'학습완료 best_iter={m.best_iteration}')

# 스무딩 맵도 함께 저장 (추론시 재계산 필요)
smap1 = pd.DataFrame({'k': pd.Series(list(zip(X_full['season'].astype(int), X_full['tm_matched'].astype(int)))),
                       'y': y}).groupby('k')['y'].agg(['sum', 'count'])
smap1 = ((smap1['sum'] + K_SHR * y.mean()) / (smap1['count'] + K_SHR)).to_dict()
smap2 = pd.DataFrame({'k': pd.Series(list(zip(X_full['cat_game_type'].astype(int), X_full['tm_lown_flag'].astype(int)))),
                       'y': y}).groupby('k')['y'].agg(['sum', 'count'])
smap2 = ((smap2['sum'] + K_SHR * y.mean()) / (smap2['count'] + K_SHR)).to_dict()

joblib.dump(dict(model=m, feat_order=FEAT_XU, raw_cols=RAW, smap_season_tmm=smap1,
                  smap_gtype_lown=smap2, g_all=float(y.mean()), k_shr=K_SHR),
            'dev/xgbunused_production.pkl')
log('저장 완료: dev/xgbunused_production.pkl')
