"""공식 실패유형 2번(크게 벗어난 볼) 조건부 분류기 g(x).
train<=2023의 CLS5==2(nd&ball) 행만으로 학습, target=y(그 부분집합 안에서는
"제구된 볼(1)" vs "크게 벗어난 볼(0)"과 동치). 프로덕션 관례(half_life=2.0) 유지.

검증:
 1) g(x) 단독 성능 (진짜 CLS2 행에서만, out-of-sample)
 2) mc5 11-class 디코더에서 ball 관련 인덱스(0,1,2)의 상수 E[y|c]를 g(x)로 교체
 3) v88 전체 블렌드 기준 H1<->H2 정직검증
"""
import numpy as np, pandas as pd, joblib, sys, time
sys.stdout.reconfigure(encoding='utf-8')
from catboost import CatBoostClassifier
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
CLS5 = np.load('dev/cls5_labels.npy')
unc = 0.249807

tr = season <= 2023
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)

# ---- g(x) 학습: train<=2023 & CLS5==2 만 ----
ball_tr = tr & (CLS5 == 2)
log(f'학습표본(train<=2023, nd&ball) n={ball_tr.sum():,}  성공률={y[ball_tr].mean():.4f}')
Xb = X.loc[ball_tr]; yb = y[ball_tr]
recency = 0.5 ** ((2023 - season[ball_tr].astype(float)) / 2.0)

n_es = int(len(Xb) * 0.92)
order = np.arange(len(Xb))  # featcache는 이미 row_num 순 정렬돼있음(세션 확립 전제)
ti, ei = order[:n_es], order[n_es:]

g_model = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                             loss_function='Logloss', verbose=False, random_seed=42,
                             min_data_in_leaf=200, early_stopping_rounds=50)
g_model.fit(Xb.iloc[ti], yb[ti], sample_weight=recency[ti],
           eval_set=(Xb.iloc[ei], yb[ei]))
log(f'g(x) 학습완료 best_iter={g_model.get_best_iteration()}')

# ---- (1) g(x) 단독 성능: 2024의 진짜 CLS2 행에서만 ----
ball_va = va & (CLS5 == 2)
g_pred_ballonly = np.clip(g_model.predict_proba(X.loc[ball_va])[:, 1], 0, 1)
y_ballonly = y[ball_va]
const_pred = np.full(ball_va.sum(), y[ball_tr].mean())
bs_g = np.mean((g_pred_ballonly - y_ballonly) ** 2)
bs_const = np.mean((const_pred - y_ballonly) ** 2)
unc_ball = y[ball_tr].mean() * (1 - y[ball_tr].mean())
print()
print(f'=== (1) g(x) 단독 (진짜 CLS2 행, n={ball_va.sum():,}) ===')
print(f'  상수(현행)  Brier={bs_const:.5f}  BSS={1e5*(1-bs_const/unc_ball):.1f}')
print(f'  g(x)        Brier={bs_g:.5f}  BSS={1e5*(1-bs_g/unc_ball):.1f}')
print(f'  개선 = {1e5*(1-bs_g/unc_ball) - 1e5*(1-bs_const/unc_ball):+.1f}')
