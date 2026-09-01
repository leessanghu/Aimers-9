"""idea99 2,3단계: g(x)를 11-class 디코더의 ball 인덱스(0,1,2)에 넣고
v88 전체 블렌드 기준 H1<->H2 정직검증."""
import numpy as np, pandas as pd, joblib, sys, time
sys.stdout.reconfigure(encoding='utf-8')
from catboost import CatBoostClassifier
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
CLS5 = np.load('dev/cls5_labels.npy')
PT = np.load('dev/pitchtype_labels.npy')
unc = 0.249807

tr = season <= 2023
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)

ball_tr = tr & (CLS5 == 2)
Xb = X.loc[ball_tr]; yb = y[ball_tr]
recency = 0.5 ** ((2023 - season[ball_tr].astype(float)) / 2.0)
n_es = int(len(Xb) * 0.92)
order = np.arange(len(Xb))
ti, ei = order[:n_es], order[n_es:]

g_model = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                             loss_function='Logloss', verbose=False, random_seed=42,
                             min_data_in_leaf=200, early_stopping_rounds=50)
g_model.fit(Xb.iloc[ti], yb[ti], sample_weight=recency[ti], eval_set=(Xb.iloc[ei], yb[ei]))
log(f'g(x) 학습완료 best_iter={g_model.get_best_iteration()}')

g_pred_va = np.clip(g_model.predict_proba(X.loc[va])[:, 1], 0, 1)

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
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
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)

cls_tr = CLS5[tr]; y_tr = y[tr]
succ_const = np.array([y_tr[cls_tr == c].mean() if (cls_tr == c).sum() > 0 else y_tr.mean() for c in range(5)])
# 11-class 순서: nd-class는 (cls5-2)*3+pt, middle=9, reverse=10
succ11_const = np.zeros(11)
for c5 in (2, 3, 4):
    for pt in range(3):
        idx = (c5 - 2) * 3 + pt
        m = (cls_tr == c5)
        succ11_const[idx] = y_tr[m].mean() if m.sum() > 0 else y_tr.mean()
succ11_const[9] = 0.0; succ11_const[10] = 0.0

mc5_const = np.clip(P11 @ succ11_const, 0, 1)

# g(x) 버전: ball 관련 인덱스 0,1,2 (nd&ball x 3개 pitch type) 를 전부 g(x)로 교체
succ11_g_base = succ11_const.copy()
decode_matrix = np.tile(succ11_g_base, (len(g_pred_va), 1))  # (n, 11)
decode_matrix[:, 0] = g_pred_va
decode_matrix[:, 1] = g_pred_va
decode_matrix[:, 2] = g_pred_va
mc5_g = np.clip((P11 * decode_matrix).sum(axis=1), 0, 1)

allm = np.ones(len(yv), bool)
print(f'mc5 단독: 상수={sc(mc5_const, allm):.2f}  g(x)버전={sc(mc5_g, allm):.2f}  ({sc(mc5_g,allm)-sc(mc5_const,allm):+.2f})')
print()

risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))


def full(mc5_pred):
    H2 = dict(H); H2['mc5'] = mc5_pred
    raw = sum(W[k] * H2[k] for k in H2)
    return np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)


f_const = full(mc5_const)
f_g = full(mc5_g)
print('=== v88 전체 블렌드 기준 ===')
print(f'  상수(현행) = {sc(f_const, allm):.2f}')
print(f'  g(x)버전   = {sc(f_g, allm):.2f}   ({sc(f_g,allm)-sc(f_const,allm):+.2f})')
print()

H1 = mth <= 6; H2m = ~H1
print('=== H1<->H2 (g(x)는 train<=2023 고정, 분할과 무관) ===')
for tag, m in [('H1', H1), ('H2', H2m)]:
    d = sc(f_g, m) - sc(f_const, m)
    print(f'  {tag}: 상수={sc(f_const, m):.2f}  g(x)={sc(f_g, m):.2f}  ({d:+.2f})')
