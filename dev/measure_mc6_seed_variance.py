"""mc6의 시드분산(sigma^2)을 직접 측정. fold A에서 seed=7로 하나 더 학습해서
기존 seed=42(dev/cache_mc6head_A.npy)와 비교.

sigma^2 = E[(p1-p2)^2] / 2   (두 시드가 iid라는 가정)
w=0.48에서 K=1 -> K=무한대 이론상한 = K_const * w^2 * sigma^2
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807
K_const = 1e5 / B_
W_MC6 = 0.48

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
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

cls = np.full(n, -1, np.int64)
cls[valid & (mid > 0.5)] = 0
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y == 0)] = 2
cls[nd & (y == 1) & (ball > 0.5)] = 3
cls[nd & (y == 1) & (strike > 0.5)] = 4
cls[nd & (y == 1) & (inplay > 0.5)] = 5
SUCC = [3, 4, 5]

upto, vs = 2023, 2024
tr = (season <= upto) & (cls >= 0)
va = season == vs
w = 0.5 ** ((upto - season[tr]) / 2.0)
n_es = int(tr.sum() * 0.92)

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=6, early_stopping_rounds=50,
          random_seed=7)

log('mc6 fold A, seed=7 학습...')
m = CatBoostClassifier(**CB)
m.fit(X.loc[tr].iloc[:n_es], cls[tr][:n_es], sample_weight=w[:n_es],
      eval_set=(X.loc[tr].iloc[n_es:], cls[tr][n_es:]))
proba = m.predict_proba(X.loc[va])
p_seed7 = np.clip(proba[:, SUCC].sum(axis=1), 0, 1)
log(f'학습완료 best_iter={m.best_iteration_}')
np.save('dev/cache_mc6head_A_seed7.npy', p_seed7)

p_seed42 = np.load('dev/cache_mc6head_A.npy')
diff = p_seed7 - p_seed42
sigma2 = float(np.mean(diff ** 2)) / 2.0
print(f'\n=== 시드분산 측정 (fold A) ===')
print(f'  sd(seed7-seed42) = {np.std(diff):.5f}')
print(f'  sigma^2(1시드 분산) = {sigma2:.4e}   sigma={np.sqrt(sigma2):.5f}')

# K=1 -> K=무한대 이론상한
ceiling_inf = K_const * W_MC6 ** 2 * sigma2
print(f'\n  w={W_MC6} 에서 K=1->K=무한대 이론상한 = {ceiling_inf:+.2f}점')
for K_target in (2, 3, 5, 10):
    frac = 1 - 1 / K_target
    print(f'  K={K_target}시드면 이론상한의 {frac*100:.0f}% 회수 = {ceiling_inf*frac:+.2f}점')

yv = y[va]
H = dict(
    base=np.mean([np.load(f'dev/phase90_cache/A_base_{m_}.npy') for m_ in ('d6', 'd8', 'sub')], axis=0),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/A_core_{m_}.npy')) *
                    np.load(f'dev/phase90_cache/A_snc_{m_}.npy') for m_ in ('d6', 'd8')], axis=0),
    multires=np.mean([np.load(f'dev/idea13_cache/A_multires_s{s}.npy') for s in (42, 7)], axis=0),
    ordinal=np.mean([np.load(f'dev/idea13_cache/A_ordinal_s{s}.npy') for s in (42, 7)], axis=0),
    midother=np.mean([np.load(f'dev/idea46_cache/A_midother_s{s}.npy') for s in (42, 7)], axis=0),
    condball=np.mean([np.load(f'dev/idea54_cache/A_cond_ball_s{s}.npy') for s in (42, 7)], axis=0),
    countresid=np.mean([np.load(f'dev/idea54_cache/A_count_resid_s{s}.npy') for s in (42, 7)], axis=0),
    future50=np.mean([np.load(f'dev/idea54_cache/A_future50_multi_s{s}.npy') for s in (42, 7)], axis=0),
)
W = {k: float(v95[f'{k}_weight']) for k in H}
t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
print(f'\n=== 참고: seed42 vs seed7 단독 BSS (안정성 확인) ===')
print(f'  seed42 BSS = {sc(p_seed42):.1f}   seed7 BSS = {sc(p_seed7):.1f}   '
      f'평균(2시드) BSS = {sc((p_seed42+p_seed7)/2):.1f}')
log('완료')
