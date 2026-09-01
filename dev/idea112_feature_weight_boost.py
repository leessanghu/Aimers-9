"""전체 162피처는 유지하되, 압박/상황 피처(16개)의 CatBoost feature_weights를 키워서
분기 우선순위를 강제로 높임. idea111(좁힌 피처셋, 신호없음)과 달리 다른 피처와의
조합까지 살아있는 상태에서 압박신호가 부각되는지 확인."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
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

ALL_FEATS = list(X.columns)
tr = season <= 2023; va = season == 2024
w = 0.5 ** ((2023 - season) / 2.0)
ti_all = np.where(tr)[0]; n_es = int(len(ti_all) * 0.92)
ti, ei = ti_all[:n_es], ti_all[n_es:]
yv = y[va]
sc = lambda p_, mask: 1e5 * (1 - np.mean((np.clip(p_[mask], 0, 1) - yv[mask]) ** 2) / unc)
allm = np.ones(len(yv), bool)

def train(boost):
    fw = [boost if c in PRESSURE_FEATS else 1.0 for c in ALL_FEATS]
    m = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            loss_function='Logloss', verbose=False, random_seed=42,
                            min_data_in_leaf=200, early_stopping_rounds=50,
                            feature_weights=fw)
    m.fit(X.iloc[ti][ALL_FEATS], y[ti], sample_weight=w[ti],
          eval_set=(X.iloc[ei][ALL_FEATS], y[ei]))
    p = np.clip(m.predict_proba(X.loc[va, ALL_FEATS])[:, 1], 0, 1)
    return p, m

log('baseline(boost=1.0, 전부 동일가중) 학습...')
p_base, m_base = train(1.0)
log(f'baseline 단독 = {sc(p_base, allm):.2f}  best_iter={m_base.get_best_iteration()}')

log('boost=5.0 학습...')
p_b5, m_b5 = train(5.0)
log(f'boost=5.0 단독 = {sc(p_b5, allm):.2f}  best_iter={m_b5.get_best_iteration()}')

log('boost=15.0 학습...')
p_b15, m_b15 = train(15.0)
log(f'boost=15.0 단독 = {sc(p_b15, allm):.2f}  best_iter={m_b15.get_best_iteration()}')

v88_final = np.load('dev/cache_v88_final_2024.npy')
mth = X.loc[va, 'game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1

def h1h2(p_, tag):
    d = p_ - p_.mean()
    resid = yv - v88_final
    gains = []
    for fit_m, ev_m, t2 in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
        C = np.mean(d[fit_m] * resid[fit_m]); V = np.mean(d[fit_m] ** 2)
        a_star = C / V if V > 1e-12 else 0.0
        blend = v88_final.copy(); blend[ev_m] = v88_final[ev_m] + a_star * d[ev_m]
        g = sc(blend, ev_m) - sc(v88_final, ev_m)
        gains.append((a_star, g))
    print(f'{tag}: H1->H2 a*={gains[0][0]:+.4f} 이득={gains[0][1]:+.2f}  |  H2->H1 a*={gains[1][0]:+.4f} 이득={gains[1][1]:+.2f}  평균={np.mean([g for _,g in gains]):+.2f}')

print()
h1h2(p_base, 'boost=1.0 (baseline)')
h1h2(p_b5, 'boost=5.0')
h1h2(p_b15, 'boost=15.0')

print('\n=== boost=15.0 피처중요도 Top15 (압박피처가 올라왔는지) ===')
imp = m_b15.get_feature_importance()
order = np.argsort(imp)[::-1][:15]
for i in order:
    tag = '*' if ALL_FEATS[i] in PRESSURE_FEATS else ' '
    print(f'  {tag} {ALL_FEATS[i]:32s} {imp[i]:.3f}')
log('완료')
