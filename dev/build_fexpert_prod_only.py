"""F전문가 프로덕션만 속성 학습(게이트 생략, 블라인드 프로브용). ~8-12분."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y_all = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95a['feature_order'])
X = X_df[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
test_gt = pd.read_csv('data/test.csv', encoding='utf-8-sig', usecols=['game_type'])
r_value = test_gt['game_type'].iloc[0]
is_F = (df['game_type'] != r_value).to_numpy()

df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
order = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(n); lab[order] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate'); mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
cls = np.full(n, -1, dtype=np.int64)
cls[valid & (mid > 0.5)] = 0
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y_all == 0)] = 2
cls[nd & (y_all == 1) & (ball > 0.5)] = 3
cls[nd & (y_all == 1) & (strike > 0.5)] = 4
cls[nd & (y_all == 1) & (inplay > 0.5)] = 5
SUCC = [3, 4, 5]

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=6, early_stopping_rounds=50,
          random_seed=42)
tr = (cls >= 0) & is_F
w = 0.5 ** ((2024.0 - season[tr]) / 2.0)
Xtr, ctr = X.loc[tr], cls[tr]
n_es = int(len(Xtr) * 0.92)
log(f'F전문가 프로덕션 학습행 {tr.sum():,}')
m = CatBoostClassifier(**CB)
m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=w[:n_es],
      eval_set=(Xtr.iloc[n_es:], ctr[n_es:]))
log(f'완료 best_iter={m.best_iteration_}')
joblib.dump(dict(model=m, feat_order=FEAT, succ_classes=SUCC, r_value=r_value),
            'dev/fexpert_prod.pkl')
log('저장: dev/fexpert_prod.pkl')
