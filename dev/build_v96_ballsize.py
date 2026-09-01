"""v96 = v95 + 물리기반 '크게벗어난볼' 보정.
g2(x): trackman 릴리스일관성+무브먼트(34) + 상황(7) = 41피처, CLS5==2(nd&ball) 행에서
train 전체(2019-2024)로 학습. alpha=0.20 (H1/H2 평균 0.14보다 다소 크게, 검증된 방향 유지)."""
import numpy as np, pandas as pd, joblib, sys, time
sys.stdout.reconfigure(encoding='utf-8')
from catboost import CatBoostClassifier
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
CLS5 = np.load('dev/cls5_labels.npy')

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
tm_feats = [f for f in v88['feature_order'] if f.startswith('tm_')]
extra_feats = [f for f in ['count_state', 'balls_before', 'strikes_before', 'x_count_pressure',
                           'pitcher_hand', 'batter_hand', 'same_hand'] if f in X.columns]
PHYS_FEATS = tm_feats + extra_feats
log(f'피처 {len(PHYS_FEATS)}개')

ball_all = (CLS5 == 2)
Xb = X.loc[ball_all, PHYS_FEATS]; yb = y[ball_all]
recency = 0.5 ** ((2024 - season[ball_all].astype(float)) / 2.0)
log(f'학습표본 n={ball_all.sum():,}  성공률={yb.mean():.4f}')

n_es = int(len(Xb) * 0.92)
order = np.arange(len(Xb))
ti, ei = order[:n_es], order[n_es:]

g2 = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                        loss_function='Logloss', verbose=False, random_seed=42,
                        min_data_in_leaf=200, early_stopping_rounds=50)
g2.fit(Xb.iloc[ti], yb[ti], sample_weight=recency[ti], eval_set=(Xb.iloc[ei], yb[ei]))
log(f'g2(x) 학습완료 best_iter={g2.get_best_iteration()}')

# RNG 제거
_RNG_TYPES = ("Generator", "BitGenerator", "RandomState")
def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 8 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, "__dict__"):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG_TYPES:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            strip_rng(v, seen, depth + 1)
strip_rng(g2)

const_ball = float(yb.mean())  # E[y|CLS5==2], train 전체 기준 상수

# center 계산 (train 전체에서, Rule4 안전 - 실제 배포시 test에 적용될 상수)
g2_pred_all = np.clip(g2.predict_proba(X[PHYS_FEATS])[:, 1], 0, 1)
P11 = None  # 프로덕션 mc5_model로 직접 계산해야 함 (전체 데이터용 캐시 없음)
mc5_model = v88['mc5_model']
proba_all = mc5_model.predict_proba(X)
p_ball_all = proba_all[:, [0, 1, 2]].sum(axis=1)
signal_all = p_ball_all * (g2_pred_all - const_ball)
center = float(signal_all.mean())
log(f'const_ball={const_ball:.4f}  center={center:.6f}')

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v96 = dict(v95)
v96['ballsize_model'] = g2
v96['ballsize_feats'] = PHYS_FEATS
v96['ballsize_const'] = const_ball
v96['ballsize_center'] = center
v96['ballsize_alpha'] = 0.20
joblib.dump(v96, 'submit/model/model_artifacts_v96.pkl')
log(f'v96 저장 완료 ({time.time()-t0:.0f}s)')
