"""v117 = v95 + mc6(s1=0.48, 기존 최적) + strk_linear(s2=0.10, 신규 프로브).

strk_linear 근거:
  raw 포렌식: 0연속실패 0.5434 / 1연속 0.5197 / 2연속 0.5000 / 3연속 0.4831
              / 4연속 0.4700 / 5+연속 0.4474  -> 9.6%p 단조격차
  fold A/C: rho -0.00519 / -0.01744 (부호일치, 5개 인코딩 중 foldA 최대)
  mc6와 d벡터 상관 0.248 -> 거의 독립축
[주의] 로컬 rho는 반정보임이 확정됨([[probe-first-methodology]]). 여기선 축 선택
       근거로만 쓰고, 실제 크기는 실측 프로브로 잰다.

보조타겟: sA = clip(연속실패, 0, 10) / 10   (fold 실험의 strk_linear와 동일)
Rule4: 연속실패는 as-of 카운터 차분으로 학습데이터에서만 복원. 추론은 head0(y)만.

s2=0.10은 프로브용. 관측 후 A2를 역산해서 mc6와 결합 최적화한다.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

S_MC6 = 0.48
S_STRK = 0.10

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v112 = joblib.load('submit/model/model_artifacts_v112.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
o = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()

same_prev = np.zeros(len(o), dtype=bool)
same_prev[1:] = (pid[o][1:] == pid[o][:-1])
dn_ord = np.full(len(o), np.nan)
dn_ord[1:] = n_[o][1:] - n_[o][:-1]
valid_prev_ord = same_prev & (dn_ord == 1)

yo = y[o]
streak_ord = np.zeros(len(o))
cur = 0.0
for i in range(len(o)):
    if not valid_prev_ord[i]:
        cur = 0.0
    streak_ord[i] = cur
    cur = 0.0 if yo[i] == 1 else cur + 1
streak_raw = np.empty(n); streak_raw[o] = streak_ord
okm = np.full(n, False); okm[o] = valid_prev_ord
aux = np.where(okm, np.clip(streak_raw, 0, 10) / 10.0, np.nan)
log(f'연속실패 보조타겟: 유효 {np.isfinite(aux).sum():,} ({np.isfinite(aux).mean()*100:.1f}%)  '
    f'평균={np.nanmean(aux):.4f}  최대원값={np.nanmax(streak_raw):.0f}')

Ymat = np.column_stack([y.astype(np.float64), aux])
CAT = dict(iterations=1200, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50,
           random_seed=42)

w = 0.5 ** ((2024.0 - season) / 2.0)
n_es = int(n * 0.92)
log('strk_linear 전체데이터 학습...')
ts = time.time()
m = CatBoostRegressor(**CAT)
m.fit(X.iloc[:n_es], Ymat[:n_es], sample_weight=w[:n_es],
      eval_set=(X.iloc[n_es:], Ymat[n_es:]))
log(f'학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')

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


strip_rng(m)

v117 = dict(v112)   # mc6pure_model/succ_classes 승계
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
rest = 1.0 - S_MC6 - S_STRK
print(f'\n=== 가중치 (v95 원본 x {rest:.2f}) ===')
for k in HEADS:
    orig = float(v95[f'{k}_weight'])
    v117[f'{k}_weight'] = orig * rest
    print(f'  {k:12s} v95={orig:.4f} -> {orig*rest:.4f}')
v117['mc6pure_weight'] = S_MC6
v117['strk_weight'] = S_STRK
v117['strk_model'] = m
tot = sum(float(v117[f'{k}_weight']) for k in HEADS) + S_MC6 + S_STRK
print(f'  mc6pure      0.0000 -> {S_MC6:.4f}')
print(f'  strk         0.0000 -> {S_STRK:.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9
assert v117.get('mc6pure_model') is not None

joblib.dump(v117, 'submit/model/model_artifacts_v117.pkl')
log('v117 저장 완료')
