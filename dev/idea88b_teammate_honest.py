"""팀원 파이프라인을 train<=2023으로 직접 재학습(정직) -> 2024 검증 + v88과 오차상관.
pipeline.py는 추론용 FE 함수만 포함(학습코드 없음) -> 여기서 재구성:
  base_model: CatBoostClassifier(feats) target=control_success
  resid_model: CatBoostRegressor(rfeats) target=(y - inseason_success), 추정 재구성
"""
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
log(f'feats={len(feats)} rfeats={len(rfeats)} wr={wr}')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
unc = 0.249807

tr = season <= 2023
train_df = df.loc[tr].reset_index(drop=True)
log(f'train_df(<=2023) n={len(train_df):,}')

tm = P._load_tm()
tables = P.build_tables(train_df) if hasattr(P, 'build_tables') else None
if tables is None:
    # pipeline.py 내부 함수명 확인
    print('build_tables 없음 -> pipeline.py 함수 목록:', [x for x in dir(P) if not x.startswith('_')])
    raise SystemExit

df_fe = P.apply_fe(df, tm, tables)   # 전체(train+val 둘다)에 동일 테이블(<=2023 기반) 적용
tr_fe = df_fe.loc[tr].reset_index(drop=True)
yt = y[tr]

def fill(d, cols, cc):
    d = d[cols].copy()
    for c in cc:
        if c in d.columns:
            d[c] = d[c].fillna('missing')
    return d

log('base_model 학습...')
base_params = dict(iterations=497, learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
                   loss_function='Logloss', verbose=False, random_seed=42)
base_model = CatBoostClassifier(**base_params)
base_model.fit(fill(tr_fe, feats, cats), yt, cat_features=[i for i,c in enumerate(feats) if c in cats])
log('base_model 완료')

log('resid_model 학습 (target = y - inseason_success)...')
target_resid = yt - tr_fe['inseason_success'].fillna(0.5).to_numpy(np.float64)
resid_model = CatBoostRegressor(iterations=497, learning_rate=0.03, depth=6, l2_leaf_reg=6.0,
                                loss_function='RMSE', verbose=False, random_seed=42)
resid_model.fit(fill(tr_fe, rfeats, cats_r), target_resid, cat_features=[i for i,c in enumerate(rfeats) if c in cats_r])
log('resid_model 완료')

p_base = base_model.predict_proba(fill(df_fe, feats, cats))[:, 1]
base_ins = df_fe['inseason_success'].fillna(0.5).to_numpy(np.float64)
p_resid = np.clip(base_ins + resid_model.predict(fill(df_fe, rfeats, cats_r)), 0, 1)
pt = (1 - wr) * p_base + wr * p_resid

va = season == 2024
yv = y[va]
sc = lambda p_: 1e5 * (1 - np.mean((np.clip(p_, 0, 1) - yv) ** 2) / unc)
print(f'정직 재학습 teammate(무보정, raw pt) 2024 BSS = {sc(pt[va]):.1f}')
for s in (2022, 2023, 2024):
    m = season == s
    print(f'  season={s}  BSS={sc(pt[m]) if s==2024 else 1e5*(1-np.mean((np.clip(pt[m],0,1)-y[m])**2)/unc):.1f}')

# v88_final
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

pred_v = np.clip(pt[va], 0, 1)
print()
print(f'v88_final BSS = {sc(v88_final):.1f}')
err_t = pred_v - yv
err_v88 = v88_final - yv
rho = np.corrcoef(err_t, err_v88)[0, 1]
pred_rho = np.corrcoef(pred_v, v88_final)[0, 1]
print(f'corr(teammate_err, v88_err) = {rho:.4f}')
print(f'corr(teammate_pred, v88_pred) = {pred_rho:.4f}')
B1 = np.mean(err_v88 ** 2); B2 = np.mean(err_t ** 2)
thresh = np.sqrt(B1 / B2)
print(f'B1={B1:.6f}  B2={B2:.6f}  임계={thresh:.6f}  {"통과" if rho<thresh else "미달"}')
resid_v88 = yv - v88_final
d = pred_v - v88_final
C = np.mean(d * resid_v88); V = np.mean(d ** 2)
a_star = C / V if V > 1e-12 else 0.0
blend = v88_final + a_star * d
print(f'최적계수 a*={a_star:.4f}  블렌드 BSS={sc(blend):.1f}  (v88단독 대비 {sc(blend)-sc(v88_final):+.2f})')
