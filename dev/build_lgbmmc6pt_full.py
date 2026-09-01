"""lgbmmc6pt 프로덕션 전체데이터(2019-2024) 학습 + 저장. 빠른 프로브용."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from lightgbm import LGBMClassifier, early_stopping

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')
ptype = np.load('dev/recovered_pitch_type.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
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


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
middle = valid & (mid > 0.5)
reverse = valid & (rev > 0.5) & (mid < 0.5)
nd = valid & (mid < 0.5) & (rev < 0.5)
ok_pt = ptype >= 0
is_fb, is_bk, is_os = ptype == 0, ptype == 1, ptype == 2
cls = np.full(n, -1, dtype=np.int64)
cls[middle] = 0
cls[reverse] = 1
cls[nd & (y == 0)] = 2
succ_ok = nd & (y == 1) & ok_pt
cls[succ_ok & is_fb] = 3
cls[succ_ok & is_bk] = 4
cls[succ_ok & is_os] = 5
SUCC = [3, 4, 5]

LGB = dict(objective='multiclass', num_class=6, n_estimators=800, learning_rate=0.05,
           num_leaves=63, min_child_samples=100, subsample=0.9, subsample_freq=1,
           colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
           n_jobs=-1, verbose=-1)

tr = cls >= 0
w = 0.5 ** ((2024.0 - season[tr]) / 2.0)
Xtr, ctr = X.loc[tr], cls[tr]
n_es = int(len(Xtr) * 0.92)
ts = time.time()
m = LGBMClassifier(**LGB)
m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=w[:n_es],
      eval_set=[(Xtr.iloc[n_es:], ctr[n_es:])],
      callbacks=[early_stopping(50, verbose=False)])
log(f'학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')

joblib.dump(dict(model=m, feat_order=FEAT, succ_classes=SUCC), 'dev/lgbmmc6pt_production.pkl')
log('저장 완료: dev/lgbmmc6pt_production.pkl')
