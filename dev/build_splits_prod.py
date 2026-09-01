"""same_hand / two_strike 분할 전문가 프로덕션(전체데이터). 둘 다 fold A 스크리닝 통과
(z=4.3 / z=2.7, mc6split 축 직교화 후에도 생존)."""
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

sh = X_df['same_hand'].to_numpy(np.float64)
stk_b = X_df['strikes_before'].to_numpy(np.float64)
SPLITS = {
    'same_hand': (sh > 0.5).astype(np.int64),
    'two_strike': (stk_b >= 2).astype(np.int64),
}

for split_name, bucket in SPLITS.items():
    models = {}
    for b in (0, 1):
        tr = (cls >= 0) & (bucket == b)
        w = 0.5 ** ((2024.0 - season[tr]) / 2.0)
        Xtr, ctr = X.loc[tr], cls[tr]
        n_es = int(len(Xtr) * 0.92)
        log(f'[{split_name}] 버킷{b} 학습행 {tr.sum():,}')
        m = CatBoostClassifier(**CB)
        m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=w[:n_es],
              eval_set=(Xtr.iloc[n_es:], ctr[n_es:]))
        log(f'[{split_name}] 버킷{b} 완료 best_iter={m.best_iteration_}')
        models[b] = m
    joblib.dump(dict(model_0=models[0], model_1=models[1], feat_order=FEAT,
                      succ_classes=SUCC, split_feature=('same_hand' if split_name == 'same_hand'
                                                         else 'strikes_before')),
                f'dev/{split_name}_production.pkl')
    log(f'[{split_name}] 저장 완료')
log('전체 완료')
