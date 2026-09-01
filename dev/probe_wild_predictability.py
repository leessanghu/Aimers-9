"""mc6을 쪼개기 전에: 'wild'(공식실패2) 하나만 따로 예측가능한지 싸게 진단.
성공 3분할(succ_ball/strk/play)도 마찬가지로 따로 진단.
학습 30분짜리 CatBoost multi-task 말고, 가벼운 HGB 단일타겟으로 5분 안에 신호유무만 본다.
"""
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

df = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['pitcher_id', 'row_id', 'asof_pitcher_n'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
order = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])
raw_full = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                       usecols=['asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate'])


def diff_label(col):
    c = np.round(raw_full[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(n); lab[order] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
nd = valid & (mid < 0.5) & (rev < 0.5)

is_wild = np.where(nd, (y == 0).astype(np.float64), np.nan)   # nd 중에서 실패=wild
succ_sub = np.full(n, -1, dtype=np.int64)   # nd & y==1 에서만: 0=ball 1=strike 2=play
succ_sub[nd & (y == 1) & (ball > 0.5)] = 0
succ_sub[nd & (y == 1) & (strike > 0.5)] = 1
succ_sub[nd & (y == 1) & (inplay > 0.5)] = 2

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
    log(f'[{tag}] is_wild (nd 중 wild vs success) 진단...')
    quick_auc(tr, is_wild, va, is_wild, 'is_wild (2-class)')

    log(f'[{tag}] 성공-존밖(ball) vs 성공-나머지(strike+play) 진단...')
    is_succball = np.where(succ_sub >= 0, (succ_sub == 0).astype(np.float64), np.nan)
    quick_auc(tr, is_succball, va, is_succball, 'succ_ball vs 나머지')

    log(f'[{tag}] 성공-존안(strike) vs 성공-인플레이(play) 진단 (ball 제외)...')
    m2 = succ_sub != 0
    is_strk = np.where((succ_sub >= 0) & m2, (succ_sub == 1).astype(np.float64), np.nan)
    quick_auc(tr & m2, is_strk, va & m2, is_strk, 'strike vs play')

log('\n완료. AUC가 0.55 이하면 그 서브타겟은 학습불가(노이즈) -> mc6에서 빼야 함.')
log('AUC가 0.60+면 그 서브타겟만 분리해서 별도 헤드로 재검증할 가치 있음.')
