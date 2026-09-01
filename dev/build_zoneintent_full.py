"""xgb_zoneintent 프로덕션: 전체데이터(2019-2024) 학습. fold A z=2.6 (미실측 최강 후보).
5클래스: middle/reverse/wild/succ_out존/succ_in존. P(성공)=class3+4.
로컬방향 음수(s*=-0.125)였으나 부호 불신 - 소량 프로브로 측정 예정."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import xgboost as xgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X_df[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')
zi = np.load('dev/recovered_zone_intent.npy')

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
ball = call[:, 0]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
middle = valid & (mid > 0.5)
reverse = valid & (rev > 0.5) & (mid < 0.5)
nd = valid & (mid < 0.5) & (rev < 0.5)
succ = nd & (y == 1)
zi_ok = succ & np.isfinite(zi)
cls = np.full(n, -1, dtype=np.int64)
cls[middle] = 0
cls[reverse] = 1
cls[nd & (y == 0)] = 2
cls[zi_ok & (zi < 0.5)] = 3
cls[zi_ok & (zi >= 0.5)] = 4
SUCC = [3, 4]
log('클래스 분포: ' + '  '.join(f'{c}:{(cls==c).mean()*100:.1f}%' for c in range(5)))

PARAMS = dict(n_estimators=800, learning_rate=0.05, max_depth=6, min_child_weight=20,
              subsample=0.9, colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0,
              max_bin=256, tree_method='hist', random_state=42, n_jobs=-1,
              early_stopping_rounds=50, objective='multi:softprob', num_class=5,
              eval_metric='mlogloss')

tr = cls >= 0
w = 0.5 ** ((2024.0 - season[tr]) / 2.0)
Xtr, ctr = X.loc[tr], cls[tr]
n_es = int(len(Xtr) * 0.92)
ts = time.time()
m = xgb.XGBClassifier(**PARAMS)
m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=w[:n_es],
      eval_set=[(Xtr.iloc[n_es:], ctr[n_es:])], verbose=False)
log(f'학습완료 best_iter={m.best_iteration} ({time.time()-ts:.0f}s)')

joblib.dump(dict(model=m, feat_order=FEAT, succ_classes=SUCC), 'dev/zoneintent_production.pkl')
log('저장 완료: dev/zoneintent_production.pkl')
