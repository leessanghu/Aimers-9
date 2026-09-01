"""middle/reverse의 판정축 세분할이 학습가능한지 빠르게 진단(HGB, 5분).
mc8 사전에 썼던 것과 동일 절차. fold A/C 재현 + AUC."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64).fillna(0.0)
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
o = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[o[:-1]] = (pid[o][1:] == pid[o][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[o]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[o]] = np.nan
    lab = np.empty(n); lab[o] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
middle = valid & (mid > 0.5)
reverse = valid & (rev > 0.5) & (mid < 0.5)

# 이진타겟: middle 도메인에서 ball vs 나머지, strike vs 나머지
mid_ball = np.where(middle, (ball > 0.5).astype(np.float64), np.nan)
mid_strk = np.where(middle, (strike > 0.5).astype(np.float64), np.nan)
rev_ball = np.where(reverse, (ball > 0.5).astype(np.float64), np.nan)
rev_strk = np.where(reverse, (strike > 0.5).astype(np.float64), np.nan)

HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=300, learning_rate=0.05,
           l2_regularization=5.0, early_stopping=True, validation_fraction=0.08,
           n_iter_no_change=20, random_state=42)


def quick_auc(mask_tr, target_tr, mask_va, target_va, name):
    m_tr = mask_tr & np.isfinite(target_tr)
    m_va = mask_va & np.isfinite(target_va)
    clf = HistGradientBoostingClassifier(**HGB)
    clf.fit(X.loc[m_tr], target_tr[m_tr])
    p = clf.predict_proba(X.loc[m_va])[:, 1]
    auc = roc_auc_score(target_va[m_va], p)
    base = target_va[m_va].mean()
    print(f'  {name:<20} n_tr={m_tr.sum():>9,}  n_va={m_va.sum():>8,}  '
          f'base_rate={base:.4f}  AUC={auc:.4f}')
    return auc


for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    tr = season <= upto
    va = season == vs
    log(f'[{tag}] middle&ball vs 나머지(middle내)...')
    quick_auc(tr, mid_ball, va, mid_ball, 'mid: ball vs 나머지')
    log(f'[{tag}] middle&strike vs 나머지(middle내)...')
    quick_auc(tr, mid_strk, va, mid_strk, 'mid: strike vs 나머지')
    log(f'[{tag}] reverse&ball vs 나머지(reverse내)...')
    quick_auc(tr, rev_ball, va, rev_ball, 'rev: ball vs 나머지')
    log(f'[{tag}] reverse&strike vs 나머지(reverse내)...')
    quick_auc(tr, rev_strk, va, rev_strk, 'rev: strike vs 나머지')

log('완료. AUC 0.55이하=노이즈, 0.60+=신호있음(오늘 wild/succball 기준 0.58~0.63이 유의했음).')
