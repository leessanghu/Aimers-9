"""접근② 물리기반: trackman 릴리스일관성+무브먼트 피처(34개)만으로
'크게 벗어난 볼' g2(x) 학습. 투수실력(ability/inseason) 축을 의도적으로 배제해서
base/hurdle 등 다른 헤드와 겹치지 않는 신호를 노린다.
+ count_state 등 상황피처 소수 추가(압박 상황에서 릴리스가 더 흔들릴 수 있음).
검증: 단독 -> 독립 additive 최적계수(H1<->H2) -> 월별 세분화(진짜 추세 vs 노이즈).
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
tr = season <= 2023; va = season == 2024
yv = y[va]; mth = X.loc[va, 'game_month'].to_numpy()
sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
tm_feats = [f for f in v88['feature_order'] if f.startswith('tm_')]
extra_feats = [f for f in ['count_state', 'balls_before', 'strikes_before', 'x_count_pressure',
                           'pitcher_hand', 'batter_hand', 'same_hand'] if f in X.columns]
PHYS_FEATS = tm_feats + extra_feats
log(f'물리기반 피처 {len(PHYS_FEATS)}개: {PHYS_FEATS}')

ball_tr = tr & (CLS5 == 2)
Xb = X.loc[ball_tr, PHYS_FEATS]; yb = y[ball_tr]
recency = 0.5 ** ((2023 - season[ball_tr].astype(float)) / 2.0)
n_es = int(len(Xb) * 0.92)
order = np.arange(len(Xb)); ti, ei = order[:n_es], order[n_es:]

g2 = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                        loss_function='Logloss', verbose=False, random_seed=42,
                        min_data_in_leaf=200, early_stopping_rounds=50)
g2.fit(Xb.iloc[ti], yb[ti], sample_weight=recency[ti], eval_set=(Xb.iloc[ei], yb[ei]))
log(f'g2(x) 학습완료 best_iter={g2.get_best_iteration()}')

# (1) 단독 성능 (진짜 CLS2 행)
ball_va = va & (CLS5 == 2)
g2_ballonly = np.clip(g2.predict_proba(X.loc[ball_va, PHYS_FEATS])[:, 1], 0, 1)
y_ballonly = y[ball_va]
const_pred = np.full(ball_va.sum(), y[ball_tr].mean())
unc_ball = y[ball_tr].mean() * (1 - y[ball_tr].mean())
bs_g2 = np.mean((g2_ballonly - y_ballonly) ** 2)
bs_const = np.mean((const_pred - y_ballonly) ** 2)
print()
print(f'=== (1) g2(x) 단독 (물리기반, n={ball_va.sum():,}) ===')
print(f'  상수      BSS={1e5*(1-bs_const/unc_ball):.1f}')
print(f'  g2(x)     BSS={1e5*(1-bs_g2/unc_ball):.1f}   ({1e5*(1-bs_g2/unc_ball)-1e5*(1-bs_const/unc_ball):+.1f})')

# feature importance 확인 (뭘 실제로 쓰는지)
imp = g2.get_feature_importance()
top = sorted(zip(PHYS_FEATS, imp), key=lambda t: -t[1])[:10]
print('  주요 피처:', [(f'{n}:{v:.1f}') for n, v in top])
print()

# (2) v88 기준 신호 구성 + 독립 additive 최적계수
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
p = 'A'
H = dict(
    base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/{p}_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/{p}_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42, 7)]),
)
P11 = np.load('dev/idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
raw = sum(W[k] * H[k] for k in H)
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
v88_final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

cls_tr = CLS5[tr]; y_tr = y[tr]
const_ball = y_tr[cls_tr == 2].mean()
g2_pred_va = np.clip(g2.predict_proba(X.loc[va, PHYS_FEATS])[:, 1], 0, 1)
p_ball_va = P11[:, [0, 1, 2]].sum(axis=1)
signal = p_ball_va * (g2_pred_va - const_ball)
resid = yv - v88_final

H1 = mth <= 6; H2m = ~H1
print('=== (2) 독립 additive 최적계수 (H1<->H2) ===')
gains = []
for fit_m, ev_m, tag in [(H1, H2m, 'H1->H2'), (H2m, H1, 'H2->H1')]:
    center = signal[fit_m].mean()
    cc = signal - center
    C = np.mean(cc[fit_m] * resid[fit_m]); V = np.mean(cc[fit_m] ** 2)
    a = C / V if V > 1e-12 else 0.0
    adj = v88_final.copy(); adj[ev_m] = v88_final[ev_m] + a * cc[ev_m]
    g = sc(adj, ev_m) - sc(v88_final, ev_m)
    gains.append(g)
    print(f'  {tag}: a*={a:.4f}  eval이득={g:+.2f}')
print(f'  평균 = {np.mean(gains):+.2f}')
print()

print('=== (3) 월별 corr(신호, 잔차) - 노이즈 vs 추세 확인 ===')
for m in sorted(np.unique(mth)):
    mm = mth == m
    if mm.sum() < 500:
        continue
    r = np.corrcoef(signal[mm], resid[mm])[0, 1]
    print(f'  {int(m):2d}월  n={mm.sum():>7,}  corr={r:+.4f}')
