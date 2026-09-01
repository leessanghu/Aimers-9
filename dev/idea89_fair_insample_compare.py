"""팀원 모델(전체데이터 학습)과 우리 v88(전체데이터 학습)을 '같은 조건'(둘 다
in-sample)에서 비교. 진짜 unseen 데이터는 아니지만, 최소한 공정한 비교는 됨.
v88은 submit/script.py의 실제 build_features + 실제 저장된 모델로 train.csv를 돌린다."""
import sys, time, importlib.util
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'dev/teammate_v1')
import numpy as np, pandas as pd, joblib
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
unc = 0.249807

# ---------- 팀원 예측 (전체데이터 학습 pkl 그대로) ----------
import pipeline as P
art = joblib.load('dev/teammate_v1/model/model_artifacts_v1_inseason_resid.pkl')
tm = P._load_tm()
df_fe = P.apply_fe(df, tm, art['tables'])
feats, cats = art['feats'], art['cats']
rfeats, cats_r = art['rfeats'], art['cats_r']

def fill(d, cols, cc):
    d = d[cols].copy()
    for c in cc:
        if c in d.columns:
            d[c] = d[c].fillna('missing')
    return d

p_base = art['base_model'].predict_proba(fill(df_fe, feats, cats))[:, 1]
base_ins = df_fe['inseason_success'].fillna(0.5).values
p_resid = np.clip(base_ins + art['resid_model'].predict(fill(df_fe, rfeats, cats_r)), 0, 1)
pt = (1 - art['wr']) * p_base + art['wr'] * p_resid
eps = 1e-6
lg = np.log(np.clip(pt, eps, 1 - eps) / (1 - np.clip(pt, eps, 1 - eps)))
Xc = np.column_stack([lg ** k for k in range(1, 6)])
z = Xc @ np.asarray(art['calib_coef']).ravel() + art['calib_intercept']
pred_teammate = np.clip(1 / (1 + np.exp(-z)), 1e-4, 1 - 1e-4)
log('팀원 예측 완료')

# ---------- 우리 v88 진짜 예측 (submit/script.py 로직 그대로, train.csv 대상) ----------
sys.path.insert(0, 'submit')
spec = importlib.util.spec_from_file_location("script", "submit/script.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
stats = v88['stats']
hgbs = v88['hgbs']; cats_v88 = v88['cats']
X_base = script.build_features(df, stats)
X_inseason = script.build_inseason_features(df, v88['inseason_stats'])
prior_rate = script.get_prior_pitcher_rate(df, v88['inseason_stats'])
X_platoon = script.build_platoon_features(df, v88['platoon_stats'], prior_rate)
X_inning = script.build_inning_features(df, v88['inning_stats'], prior_rate)
X_count = script.build_count_features(df, v88['count_stats'], prior_rate)
X_pitchtype = script.build_pitchtype_features(df, v88['pitchtype_stats'], prior_rate)
X_lastyear = script.build_lastyear_features(df, v88['lastyear_stats'])
X_volatility = script.build_volatility_features(df, v88['volatility_stats'])
X_role = script.build_role_features(df, v88['role_stats'])
base_middle_arr = np.full(len(df), v88['form_base_middle_global'])
X_form = script.build_form_features(df, X_role, X_inseason['inseason_success_smooth'].to_numpy(np.float64), base_middle_arr)
X_trackman = script.build_trackman_features(df, v88['trackman_stats'])
X_trackman_lown = script.build_trackman_lown_features(X_trackman, df['asof_pitcher_n'].to_numpy(np.float64), v88['trackman_stats']['lown_threshold'])
X_batter = script.build_batter_features(df, v88['batter_stats'])
n_end_row = script.get_n_end(df, v88['inseason_stats'])
X_inseason_full = script.build_inseason_full_features(df, v88['inseason_full_stats'], n_end_row,
    X_inseason['inseason_success_smooth'].to_numpy(np.float64), X_inseason['inseason_reverse_smooth'].to_numpy(np.float64))
X_bat_middle = script.build_batter_middle_features(df, v88['batter_split_stats'])
X_bplatoon = script.build_bplatoon_features(df, v88['batter_split_stats'])

X = pd.concat([X_base, X_inseason, X_platoon, X_inning, X_pitchtype], axis=1).astype(np.float64)
X_cross = script.add_crosses(X)
X = pd.concat([X, X_cross, X_lastyear, X_count, X_volatility, X_role, X_form, X_trackman,
               X_trackman_lown, X_batter, X_inseason_full, X_bat_middle, X_bplatoon], axis=1)
X = X[v88['feature_order']].astype(np.float64)
log(f'v88 피처 구성 완료 {X.shape}')

def _hgb_predict(m):
    if hasattr(m, 'predict_proba'):
        return m.predict_proba(X)[:, 1]
    return np.clip(m.predict(X), 0.0, 1.0)
p_hgb = np.mean([_hgb_predict(m) for m in hgbs], axis=0)
p_cat = np.mean([c.predict_proba(X)[:, 1] for c in cats_v88], axis=0)
p_ensemble = v88['w_hgb'] * p_hgb + v88['w_cat'] * p_cat
log('base(HGBxCat) 완료')

cfm = v88['core_fail_models']; snm = v88['succ_nc_models']
p_hurdle = np.mean([(1 - cm.predict_proba(X)[:, 1]) * sm.predict_proba(X)[:, 1] for cm, sm in zip(cfm, snm)], axis=0)
mr_m = v88['multires_model']; p_multires = np.clip(mr_m.predict(X), 0, 1)[:, 0]
o1, o2, o3 = v88['ordinal_stage1'], v88['ordinal_stage2'], v88['ordinal_stage3']
p_ordinal = o1.predict_proba(X)[:, 1] * o2.predict_proba(X)[:, 1] * o3.predict_proba(X)[:, 1]
mo_m = v88['midother_model']; p_midother = np.clip(mo_m.predict(X), 0, 1)[:, 0]
cb_m = v88['condball_model']; p_condball = np.clip(cb_m.predict(X), 0, 1)[:, 0]
cr_m = v88['countresid_model']; p_countresid = np.clip(cr_m.predict(X), 0, 1)[:, 0]
f5_m = v88['future50_model']; p_future50 = np.clip(f5_m.predict(X), 0, 1)[:, 0]
mc5_m = v88['mc5_model']; proba5 = mc5_m.predict_proba(X)
p_mc5 = np.clip(float(v88.get('mc5_intercept', 0.0)) + proba5 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing_m = v88['ingame_model']; p_ingame = np.clip(ing_m.predict(X), 0, 1)[:, 0]
log('전체 헤드 완료')

preds = (v88['base_weight'] * p_ensemble + v88['hurdle_weight'] * p_hurdle
        + v88['multires_weight'] * p_multires + v88['ordinal_weight'] * p_ordinal
        + v88['midother_weight'] * p_midother + v88['condball_weight'] * p_condball
        + v88['countresid_weight'] * p_countresid + v88['future50_weight'] * p_future50
        + v88['mc5_weight'] * p_mc5 + v88['ingame_weight'] * p_ingame)

risk_idx = v88.get('risk_class_idx')
if risk_idx is not None and v88.get('risk_alpha', 0) > 0:
    risk_vec = proba5[:, list(risk_idx)].sum(axis=1)
    cut = np.maximum(0.0, risk_vec - float(v88['risk_thr']))
    preds = preds - float(v88['risk_alpha']) * (cut - float(v88['risk_center']))
preds = preds + float(v88.get('level_shift', 0.0))
pred_v88_real = np.clip(preds, 0, 1)
log('v88 최종 예측 완료')

sc = lambda p_, m: 1e5 * (1 - np.mean((np.clip(p_[m], 0, 1) - y[m]) ** 2) / unc)
print()
print('=== 둘 다 in-sample (2019-2024 전체 학습, 같은 조건) ===')
for s in (2022, 2023, 2024):
    m = season == s
    print(f'  season={s}  v88실제={sc(pred_v88_real,m):8.1f}  teammate={sc(pred_teammate,m):8.1f}')
m_all = np.ones(len(y), bool)
print(f'  전체       v88실제={sc(pred_v88_real,m_all):8.1f}  teammate={sc(pred_teammate,m_all):8.1f}')
print()

err_t = pred_teammate - y
err_v = pred_v88_real - y
rho_all = np.corrcoef(err_t, err_v)[0, 1]
print(f'corr(오차, 전체행) = {rho_all:.4f}')
for s in (2022, 2023, 2024):
    m = season == s
    rho = np.corrcoef(err_t[m], err_v[m])[0, 1]
    print(f'  season={s} corr(오차)={rho:.4f}')

resid_v = y - pred_v88_real
d = pred_teammate - pred_v88_real
for s in (None, 2024):
    m = m_all if s is None else (season == s)
    C = np.mean(d[m] * resid_v[m]); V = np.mean(d[m] ** 2)
    a = C / V if V > 1e-12 else 0.0
    blend = pred_v88_real.copy()
    blend[m] = pred_v88_real[m] + a * d[m]
    tag = '전체' if s is None else f'season={s}'
    print(f'{tag}: a*={a:.4f}  블렌드={sc(blend,m):.1f}  (v88단독 {sc(pred_v88_real,m):.1f} 대비 {sc(blend,m)-sc(pred_v88_real,m):+.2f})')
