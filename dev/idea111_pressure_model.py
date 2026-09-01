"""압박/상황 raw피처만 좁혀서 새 모델 학습 (물리기반 볼사이즈 모델과 같은 전략).
score_diff, li, win_expectancy, 카운트, 아웃, 주자상황 등 ~15개만 사용해 y를 직접 예측.
H1/H2 독립계수(a*)로 부호까지 확인 - 음수면 반대신호 후보."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
unc = 0.249807

PRESSURE_FEATS = [c for c in [
    'score_diff_pitcher_team', 'score_diff_home', 'li', 'home_win_expectancy', 'away_win_expectancy',
    'outs_before', 'balls_before', 'strikes_before', 'count_state', 'num_runners_on',
    'runner_on_1b', 'runner_on_2b', 'runner_on_3b', 'x_count_pressure', 'inning', 'x_ability_x_pressure',
] if c in X.columns]
log(f'압박피처 {len(PRESSURE_FEATS)}개: {PRESSURE_FEATS}')

tr = season <= 2023; va = season == 2024
w = 0.5 ** ((2023 - season) / 2.0)
ti_all = np.where(tr)[0]; n_es = int(len(ti_all) * 0.92)
ti, ei = ti_all[:n_es], ti_all[n_es:]

m = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                        loss_function='Logloss', verbose=False, random_seed=42,
                        min_data_in_leaf=200, early_stopping_rounds=50)
m.fit(X.iloc[ti][PRESSURE_FEATS], y[ti], sample_weight=w[ti],
      eval_set=(X.iloc[ei][PRESSURE_FEATS], y[ei]))
log(f'압박모델 학습완료 best_iter={m.get_best_iteration()}')

g_pred = np.clip(m.predict_proba(X.loc[va, PRESSURE_FEATS])[:, 1], 0, 1)
yv = y[va]
sc = lambda p_, mask: 1e5 * (1 - np.mean((np.clip(p_[mask], 0, 1) - yv[mask]) ** 2) / unc)
allm = np.ones(len(yv), bool)
print(f'\n압박모델 단독 = {sc(g_pred, allm):.2f}  (상수={y[tr].mean():.4f} 기준 BSS={1e5*(1-np.mean((y[tr].mean()-yv)**2)/unc):.2f})')

imp = m.get_feature_importance()
order = np.argsort(imp)[::-1]
print('\n압박모델 피처중요도:')
for i in order:
    print(f'  {PRESSURE_FEATS[i]:28s} {imp[i]:.3f}')

v88_final = np.load('dev/cache_v88_final_2024.npy')
mth = X.loc[va, 'game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
d = g_pred - g_pred.mean()
resid = yv - v88_final
print(f'\n=== 독립 additive 최적계수(H1/H2) - 부호 확인 ===')
gains = []
for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
    C = np.mean(d[fit_m] * resid[fit_m]); V = np.mean(d[fit_m] ** 2)
    a_star = C / V if V > 1e-12 else 0.0
    blend = v88_final.copy(); blend[ev_m] = v88_final[ev_m] + a_star * d[ev_m]
    g = sc(blend, ev_m) - sc(v88_final, ev_m)
    gains.append(g)
    print(f'{tag}: a*={a_star:+.4f}  이득={g:+.2f}')
print(f'평균 이득 = {np.mean(gains):+.2f}')

print('\n=== 월별 corr(신호, 잔차) - 노이즈 vs 추세 ===')
for mo in sorted(pd.unique(mth)):
    mm = mth == mo
    if mm.sum() < 500:
        continue
    c = np.corrcoef(d[mm], resid[mm])[0, 1]
    print(f'  {mo:2.0f}월  n={mm.sum():6,}  corr={c:+.4f}')
log('완료')
