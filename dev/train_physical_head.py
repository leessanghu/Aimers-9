"""물리/커맨드 헤드: tm_* 34개 + 신규 6개 + 컨텍스트, multi-task [y, 1-middle, 1-reverse].
fold A(train<=2023 -> 2024) 검증.

해석 기준이 핵심: fold A는 '헤드'를 구조적으로 과소평가한다(v99/v101 실측이 증명).
따라서 절대값이 양수인지가 아니라, 실측에서 가치가 입증된 기존 헤드들의
leave-one-out 값과 비교했을 때 어느 위치인지를 봐야 한다."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807
K = 1e5 / B

X = pd.read_parquet('dev/featcache_X.parquet')
newf = pd.read_parquet('dev/new_tm_features.parquet')
X = pd.concat([X.reset_index(drop=True), newf.reset_index(drop=True)], axis=1)
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
cls5 = np.load('dev/cls5_labels.npy')
valid5 = cls5 >= 0
is_mid = cls5 == 0
is_rev = cls5 == 1

tr = season <= 2023
va = season == 2024
yv = y[va]
w = 0.5 ** ((2023 - season) / 2.0)

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
tm_feats = [c for c in v88['feature_order'] if c.startswith('tm_')]
NEWF = list(newf.columns)
CTX = [c for c in ['count_state', 'balls_before', 'strikes_before', 'outs_before', 'inning',
                   'pitcher_hand', 'batter_hand', 'same_hand', 'hand_matchup',
                   'x_count_pressure', 'asof_pitcher_n'] if c in X.columns]
FEATS = tm_feats + NEWF + CTX
log(f'피처 {len(FEATS)}개 = tm {len(tm_feats)} + 신규 {len(NEWF)} + 컨텍스트 {len(CTX)}')

h1 = np.where(valid5, 1.0 - is_mid.astype(np.float64), np.nan)
h2 = np.where(valid5, 1.0 - is_rev.astype(np.float64), np.nan)
Ymat = np.column_stack([y, h1, h2])

ti_all = np.where(tr)[0]
n_es = int(len(ti_all) * 0.92)
ti, ei = ti_all[:n_es], ti_all[n_es:]

CFG = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50)
log('물리/커맨드 헤드 학습...')
m = CatBoostRegressor(**CFG, random_seed=42)
m.fit(X.iloc[ti][FEATS], Ymat[ti], sample_weight=w[ti],
      eval_set=(X.iloc[ei][FEATS], Ymat[ei]))
p_new = np.clip(m.predict(X.loc[va, FEATS])[:, 0], 0, 1)
log(f'학습완료 best_iter={m.best_iteration_}')

imp = m.get_feature_importance()
order = np.argsort(imp)[::-1]
print('\n=== 물리헤드 피처중요도 Top15 (신규는 ★) ===')
for i in order[:15]:
    mark = '★' if FEATS[i] in NEWF else ' '
    print(f'  {mark} {FEATS[i]:28s} {imp[i]:.3f}')
print('  --- 신규 6개 순위 ---')
for f in NEWF:
    i = FEATS.index(f)
    print(f'    {f:28s} rank{list(order).index(i)+1:3d}  imp={imp[i]:.3f}')

# fold A 블렌드 재구성
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
keys = list(H.keys())
W = {k: float(v88[f'{k}_weight']) for k in keys}

risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))


def finalize(raw):
    return np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center']))
                   + float(v88['level_shift']), 0, 1)


base_pred = finalize(sum(W[k] * H[k] for k in keys))
sc = lambda q, msk: 1e5 * (1 - np.mean((np.clip(q[msk], 0, 1) - yv[msk]) ** 2) / B)
allm = np.ones(len(yv), bool)
mth = X.loc[va, 'game_month'].to_numpy()
M1 = mth <= 6
M2 = ~M1
resid = yv - base_pred

print(f'\nv88_final(fold A) = {sc(base_pred, allm):.2f}')
print(f'물리헤드 단독      = {sc(p_new, allm):.2f}')


def blend_gain(cand, tag):
    d = cand - base_pred
    C = np.mean(d * resid)
    V = np.mean(d ** 2)
    maxg = (C * C / V) * K if V > 1e-12 else 0.0
    gains = []
    for fit_m, ev_m in [(M1, M2), (M2, M1)]:
        Cf = np.mean(d[fit_m] * resid[fit_m])
        Vf = np.mean(d[fit_m] ** 2)
        a = Cf / Vf if Vf > 1e-12 else 0.0
        bl = base_pred.copy()
        bl[ev_m] = base_pred[ev_m] + a * d[ev_m]
        gains.append(sc(bl, ev_m) - sc(base_pred, ev_m))
    print(f'  {tag:22s} 이론최대={maxg:+7.2f}  H1->H2={gains[0]:+7.2f}  H2->H1={gains[1]:+7.2f}  평균={np.mean(gains):+7.2f}')
    return np.mean(gains)


print('\n=== 신규 물리헤드 블렌드 기여 ===')
blend_gain(p_new, '물리/커맨드 헤드')

print('\n=== [비교기준] 기존 헤드 leave-one-out (fold A) ===')
print('  * 실측(v99/v101)은 이들을 빼면 -15.66/-8.66 손해였다. 즉 fold A가 음수여도')
print('    실제 가치가 있다는 게 이미 증명된 헤드들이다.')
for k in keys:
    wsum = sum(W[j] for j in keys if j != k)
    lo = finalize(sum(W[j] / wsum * H[j] for j in keys if j != k))
    print(f'  {k:12s} 제거시 {sc(lo, allm) - sc(base_pred, allm):+7.2f}')

np.save('dev/cache_physhead_2024.npy', p_new)
log('완료 (예측 저장: dev/cache_physhead_2024.npy)')
