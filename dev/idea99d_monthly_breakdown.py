"""H1/H2 두 덩어리 대신 월별로 g(x) 신호와 잔차의 관계를 본다.
완만한 추세(리그 레짐 변화)면 월별로 부드럽게 이어질 것이고,
순수 노이즈면 월마다 들쭉날쭉할 것이다."""
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

ball_tr = tr & (CLS5 == 2)
Xb = X.loc[ball_tr]; yb = y[ball_tr]
recency = 0.5 ** ((2023 - season[ball_tr].astype(float)) / 2.0)
n_es = int(len(Xb) * 0.92)
order = np.arange(len(Xb)); ti, ei = order[:n_es], order[n_es:]
g_model = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                             loss_function='Logloss', verbose=False, random_seed=42,
                             min_data_in_leaf=200, early_stopping_rounds=50)
g_model.fit(Xb.iloc[ti], yb[ti], sample_weight=recency[ti], eval_set=(Xb.iloc[ei], yb[ei]))
log(f'g(x) 학습완료')
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
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
raw = sum(W[k] * H[k] for k in H)
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
v88_final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

cls_tr = CLS5[tr]; y_tr = y[tr]
const_ball = y_tr[cls_tr == 2].mean()
p_ball_va = P11[:, [0, 1, 2]].sum(axis=1)
signal = p_ball_va * (g_pred_va - const_ball)
resid = yv - v88_final

print('=== 월별 corr(g신호, 잔차) 및 표본수 ===')
print(f'{"월":>3s} {"n":>8s} {"신호평균":>9s} {"corr":>8s}')
for m in sorted(np.unique(mth)):
    mm = mth == m
    if mm.sum() < 500:
        continue
    r = np.corrcoef(signal[mm], resid[mm])[0, 1]
    print(f'{int(m):3d} {mm.sum():8,} {signal[mm].mean():+9.5f} {r:+8.4f}')

print()
print('=== 신규투수(rookie) 비율 월별 (레짐변화 메커니즘 확인) ===')
df = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['row_id', 'season', 'pitcher_id'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)
seen = set(df.loc[df.season <= 2023, 'pitcher_id'])
df24 = df[df.season == 2024].copy()
df24['is_new'] = ~df24.pitcher_id.isin(seen)
mth_full = X.loc[va, 'game_month'].to_numpy()
df24 = df24.reset_index(drop=True)
for m in sorted(np.unique(mth_full)):
    mm = mth_full == m
    if mm.sum() < 500:
        continue
    print(f'{int(m):3d}월  신규투수비율={df24.loc[mm,"is_new"].mean()*100:5.2f}%')
