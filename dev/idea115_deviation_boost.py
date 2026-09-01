"""'편차/트렌드' 계열(22개, 중간순위대) 피처가 압도적 피처(season, cat_game_type,
x_ability_here 등)에 묻혀서 트리가 못 읽는 건지 확인. feature_weights로 강제 부스트해서
(idea112와 동일 방법, 이번엔 다른 클러스터) fold A 정직검증 + 신규 클린 3종 세트."""
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

DEV_FEATS = [c for c in [
    'form5_success', 'form3_success', 'x_prev5_minus_career', 'ly_minus_career',
    'x_kal_minus_career', 'x_prev1_minus_prev5', 'form1_success', 'bat_middle_minus_career',
    'form_accel', 'score_diff_pitcher_team', 'form5_middle', 'form_3_minus_5', 'form3_middle',
    'inseason_middle_minus_career', 'form1_middle', 'form_reliability', 'form_1_minus_3',
    'bat_inseason_minus_career', 'diff_success_rate', 'score_diff_home', 'diff_middle_rate',
] if c in X.columns]
ALL_FEATS = list(X.columns)
log(f'편차/트렌드 피처 {len(DEV_FEATS)}개')

tr = season <= 2023; va = season == 2024
w = 0.5 ** ((2023 - season) / 2.0)
ti_all = np.where(tr)[0]; n_es = int(len(ti_all) * 0.92)
ti, ei = ti_all[:n_es], ti_all[n_es:]
yv = y[va]
B = 0.249807
sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)
allm = np.ones(len(yv), bool)


def train(boost):
    fw = [boost if c in DEV_FEATS else 1.0 for c in ALL_FEATS]
    m = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                            loss_function='Logloss', verbose=False, random_seed=42,
                            min_data_in_leaf=200, early_stopping_rounds=50, feature_weights=fw)
    m.fit(X.iloc[ti][ALL_FEATS], y[ti], sample_weight=w[ti], eval_set=(X.iloc[ei][ALL_FEATS], y[ei]))
    p = np.clip(m.predict_proba(X.loc[va, ALL_FEATS])[:, 1], 0, 1)
    return p, m


log('baseline(boost=1.0)...')
p_base, m_base = train(1.0)
log(f'baseline 단독={sc(p_base, allm):.2f}  best_iter={m_base.get_best_iteration()}')

log('boost=10.0...')
p_b10, m_b10 = train(10.0)
log(f'boost=10.0 단독={sc(p_b10, allm):.2f}  best_iter={m_b10.get_best_iteration()}')

log('boost=40.0...')
p_b40, m_b40 = train(40.0)
log(f'boost=40.0 단독={sc(p_b40, allm):.2f}  best_iter={m_b40.get_best_iteration()}')

# v88_final 기준 잔차 대비 각 후보의 순수 기여 (클린 3종: 중심화+무절편+대조군)
blend = np.load('dev/cache_v88_final_2024.npy')
resid = yv - blend
X_ = X.loc[va]
mth = X_['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
rng = np.random.RandomState(6)


def honest(dd):
    gains, coefs = [], []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        mdf = dd[fit_m].mean(); mrf = resid[fit_m].mean()
        cv = np.mean((dd[fit_m]-mdf)*(resid[fit_m]-mrf))
        vr = np.mean((dd[fit_m]-mdf)**2)
        a = cv/vr if vr > 1e-14 else 0.0
        bl = blend.copy()
        bl[ev_m] = blend[ev_m] + a*(dd[ev_m]-mdf)
        gains.append(sc(bl, ev_m) - sc(blend, ev_m))
        coefs.append(a)
    return gains, coefs


print()
for name, p in [('baseline', p_base), ('boost10', p_b10), ('boost40', p_b40)]:
    d = p - blend
    ctrl = rng.normal(0, d.std(), len(yv))
    gc, _ = honest(ctrl)
    g, c = honest(d)
    print(f'{name:10s} 대조군={np.mean(gc):+6.2f}  신호 H1->H2={g[0]:+7.2f} H2->H1={g[1]:+7.2f} '
          f'평균={np.mean(g):+7.2f}  a={c[0]:+.4f}/{c[1]:+.4f}')

print('\n=== boost=40 피처중요도 Top15 (편차/트렌드 계열이 올라왔는지) ===')
imp = m_b40.get_feature_importance()
order = np.argsort(imp)[::-1][:15]
for i in order:
    mark = '★' if ALL_FEATS[i] in DEV_FEATS else ' '
    print(f'  {mark} {ALL_FEATS[i]:32s} {imp[i]:.3f}')
log('완료')
