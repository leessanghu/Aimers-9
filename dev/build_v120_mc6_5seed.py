"""v120 = v117 구조 그대로, mc6pure만 5시드 배깅으로 교체.

측정된 시드분산(fold A): sigma^2=5.7368e-05, w=0.48에서 K=1->inf 이론상한=+5.29점.
K=5(80% 회수)로 프로덕션 재학습 -> 예상 +4.23점 (v117 1114.53 -> 약 1118.76).

시드 5개: 42,7,2024,123,777 (오늘 다른 배깅 실험과 동일 세트, v111 관례 따름).
mc6pure_model을 리스트(mc6pure_models)로 저장하고, strk/pitchtype 등 나머지는
v117 그대로 승계. script.py에서 mc6pure_models(리스트)가 있으면 5시드 평균,
없으면 mc6pure_model(단일) 폴백하도록 이미 배선된 패턴(*_models 관례)을 따른다.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

SEEDS = [42, 7, 2024, 123, 777]

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v117 = joblib.load('submit/model/model_artifacts_v117.pkl')
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
log(f'클래스 분포: ' + '  '.join(f'{c}:{(cls==c).mean()*100:.1f}%' for c in range(6)))

mask = cls >= 0
w = 0.5 ** ((2024.0 - season) / 2.0)
Xtr, ctr, wtr = X.loc[mask], cls[mask], w[mask]
n_es = int(len(Xtr) * 0.92)

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=6, early_stopping_rounds=50)

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


models = []
for s in SEEDS:
    ts = time.time()
    m = CatBoostClassifier(**CB, random_seed=s)
    m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=wtr[:n_es],
          eval_set=(Xtr.iloc[n_es:], ctr[n_es:]))
    strip_rng(m)
    models.append(m)
    log(f'seed={s} 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')

v120 = dict(v117)
v120.pop('mc6pure_model', None)
v120['mc6pure_models'] = models
v120['mc6pure_succ_classes'] = SUCC
# 가중치는 v117 그대로 유지 (mc6pure_weight=0.48, strk_weight=0.10, 나머지 동일)
joblib.dump(v120, 'submit/model/model_artifacts_v120.pkl')
log(f'v120 저장 완료 (mc6pure 5시드 배깅, 나머지 v117과 동일)')
