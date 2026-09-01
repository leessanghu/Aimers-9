"""1단계: 팀원모델 train<=2023 정직 재학습 + 예측 저장(npy) + H1/H2 양방향 검증.
v88_final은 fold A OOF 캐시(honest) 기준으로 재구성 (idea87/88과 동일)."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'dev/teammate_v1')
import numpy as np, pandas as pd, joblib
import pipeline as P
from catboost import CatBoostClassifier, CatBoostRegressor
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

art = joblib.load('dev/teammate_v1/model/model_artifacts_v1_inseason_resid.pkl')
feats, cats = art['feats'], art['cats']
rfeats, cats_r = art['rfeats'], art['cats_r']
wr = art['wr']

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
unc = 0.249807

tr = season <= 2023
train_df = df.loc[tr].reset_index(drop=True)
tm = P._load_tm()
tables = P.build_tables(train_df)
df_fe = P.apply_fe(df, tm, tables)
tr_fe = df_fe.loc[tr].reset_index(drop=True)
yt = y[tr]

def fill(d, cols, cc):
    d = d[cols].copy()
    for c in cc:
        if c in d.columns:
            d[c] = d[c].fillna('missing')
    return d

log('base_model 학습...')
base_model = CatBoostClassifier(iterations=497, learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
                                loss_function='Logloss', verbose=False, random_seed=42)
base_model.fit(fill(tr_fe, feats, cats), yt, cat_features=[i for i, c in enumerate(feats) if c in cats])
log('base_model 완료')

target_resid = yt - tr_fe['inseason_success'].fillna(0.5).to_numpy(np.float64)
resid_model = CatBoostRegressor(iterations=497, learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
                                loss_function='RMSE', verbose=False, random_seed=42)
resid_model.fit(fill(tr_fe, rfeats, cats_r), target_resid, cat_features=[i for i, c in enumerate(rfeats) if c in cats_r])
log('resid_model 완료')

p_base = base_model.predict_proba(fill(df_fe, feats, cats))[:, 1]
base_ins = df_fe['inseason_success'].fillna(0.5).to_numpy(np.float64)
p_resid = np.clip(base_ins + resid_model.predict(fill(df_fe, rfeats, cats_r)), 0, 1)
pt = np.clip((1 - wr) * p_base + wr * p_resid, 0, 1)

va = season == 2024
np.save('dev/cache_teammate_honest_2024.npy', pt[va])
log('teammate 예측(2024) 저장 완료')

# v88_final (honest, fold A OOF 캐시 기반)
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
np.save('dev/cache_v88_final_2024.npy', v88_final)
log('v88_final 저장 완료')

# H1/H2 검증
X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
mth = X.loc[meta['season'].to_numpy() == 2024, 'game_month'].to_numpy()
yv = y[va]
sc = lambda p_, m: 1e5 * (1 - np.mean((np.clip(p_[m], 0, 1) - yv[m]) ** 2) / unc)
teammate = pt[va]
H1 = mth <= 6; H2 = ~H1
resid_v = yv - v88_final
d = teammate - v88_final
print()
print(f'v88_final 단독 = {sc(v88_final, np.ones(len(yv),bool)):.1f}')
print(f'teammate 단독  = {sc(teammate, np.ones(len(yv),bool)):.1f}')
print()
gains = []
for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
    C = np.mean(d[fit_m] * resid_v[fit_m]); V = np.mean(d[fit_m] ** 2)
    w_star = C / V if V > 1e-12 else 0.0
    blend = v88_final.copy(); blend[ev_m] = v88_final[ev_m] + w_star * d[ev_m]
    g = sc(blend, ev_m) - sc(v88_final, ev_m)
    gains.append(g)
    print(f'{tag}: fit에서 구한 w*={w_star:+.4f}  eval구간 실현이득={g:+.2f}')
print(f'평균 이득 = {np.mean(gains):+.2f}')
log('완료')
