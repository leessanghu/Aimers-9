"""v113 = v95 + mc6 계층분해 3서브헤드 평균(w=-0.03, 음수가중치).

계층분해: mc6원본(6-way joint softmax, 실패)과 달리 middle/reverse는 기존
hurdle/ordinal이 이미 다루므로 제외하고, 새로 발견한 3개 이진서브타겟만 각각
독립 multi-task로 학습:
  headA_wild:   [y, is_wild]              (nd 중 실패=wild)
  headB_ball:   [y, is_succball]          (nd&성공 중 존밖 성공)
  headC_strike: [y, is_strike_amongnonball] (nd&성공&~ball 중 존안 성공)

fold A/C 검증: rho가 6/6(3서브헤드x2fold) 전부 음수로 부호 일치(오늘 유일).
클린 max-gain(fold A, 3헤드평균) = +2.50 (대조군 +1.69 초과).
=> 최적가중치는 음수(빼는 방향). w=-0.03 (|w|는 v108 XGB 선례와 동일 크기,
   SE≈0.26로 리스크 작음, fold A 로컬 s*=-0.27은 과최적화 위험 있어 참고만).

Rule4: 서브타겟 라벨은 as-of 카운터 차분 기반 학습전용, 추론은 head0(y)만 사용.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

W_NEW = -0.03

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
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


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
nd = valid & (mid < 0.5) & (rev < 0.5)

is_wild = np.where(nd, (y == 0).astype(np.float64), np.nan)
is_succball = np.where(nd & (y == 1), ball.astype(np.float64), np.nan)
notball = nd & (y == 1) & (ball < 0.5)
is_strike = np.where(notball, strike.astype(np.float64), np.nan)

TARGETS = {'headA_wild': is_wild, 'headB_ball': is_succball, 'headC_strike': is_strike}
for nm, t in TARGETS.items():
    log(f'{nm}: 유효 {np.isfinite(t).sum():,} ({np.isfinite(t).mean()*100:.1f}%)')

CAT = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50,
           random_seed=42)

_RNG = ('Generator', 'BitGenerator', 'RandomState', 'PCG64', 'MT19937', 'Philox', 'SFC64')
def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 12 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, '__dict__'):
        for k2, v2 in list(vars(obj).items()):
            if type(v2).__name__ in _RNG:
                setattr(obj, k2, None)
            else:
                strip_rng(v2, seen, depth + 1)
    elif isinstance(obj, dict):
        for k2, v2 in list(obj.items()):
            if type(v2).__name__ in _RNG:
                obj[k2] = None
            else:
                strip_rng(v2, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v2 in obj:
            strip_rng(v2, seen, depth + 1)


w = 0.5 ** ((2024.0 - season) / 2.0)
n_es = int(len(X) * 0.92)
models = {}
for nm, tgt in TARGETS.items():
    Ymat = np.column_stack([y.astype(np.float64), tgt])
    ts = time.time()
    m = CatBoostRegressor(**CAT)
    m.fit(X.iloc[:n_es], Ymat[:n_es], sample_weight=w[:n_es],
          eval_set=(X.iloc[n_es:], Ymat[n_es:]))
    strip_rng(m)
    models[nm] = m
    log(f'{nm} 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')

v113 = dict(v95)
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
print('\n=== 가중치 재배분 (기존 전부 비례조정, W_NEW 음수라 다들 소폭 상향) ===')
for k in HEADS:
    old = float(v95[f'{k}_weight'])
    new = old * (1 - W_NEW)
    v113[f'{k}_weight'] = new
    print(f'  {k:12s} {old:.4f} -> {new:.4f}')
v113['mc6hier_weight'] = W_NEW
v113['mc6hier_models'] = [models['headA_wild'], models['headB_ball'], models['headC_strike']]
tot = sum(float(v113[f'{k}_weight']) for k in HEADS) + W_NEW
print(f'  mc6hier      0.0000 -> {W_NEW:.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v113, 'submit/model/model_artifacts_v113.pkl')
log('v113 저장 완료')
