"""옛날 단일모델(v7b: 65피처, HGB만)을 정직하게(train<=2023->2024) 재학습해서
지금 v88_raw와 오차상관이 진짜로 낮은지 확인. in-sample 오염 없이."""
import sys, importlib.util, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

spec = importlib.util.spec_from_file_location("script", "submit/script.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

a = joblib.load('dev/old_models/model_artifacts_v7b.pkl')
df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()

X_base = script.build_features(df, a['stats'])
X_ins = script.build_inseason_features(df, a['inseason_stats'])
prior_rate = script.get_prior_pitcher_rate(df, a['inseason_stats'])
X_plt = script.build_platoon_features(df, a['platoon_stats'], prior_rate)
X = pd.concat([X_base, X_ins, X_plt], axis=1)[a['feature_order']].astype(np.float64)
log(f'피처 구성 완료 {X.shape}')

tr = season <= 2023
va = season == 2024
Xt, yt = X.loc[tr], y[tr]
Xv, yv = X.loc[va], y[va]
unc = 0.249807
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / unc)

hgb = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.03, max_depth=6, max_leaf_nodes=31,
    l2_regularization=5.0, min_samples_leaf=20, early_stopping=True,
    validation_fraction=0.1, n_iter_no_change=20, random_state=42)
log('학습 시작 (train<=2023만)...')
hgb.fit(Xt, yt)
log(f'학습 완료 n_iter_={hgb.n_iter_}')
pred_v7b_honest = np.clip(hgb.predict_proba(Xv)[:, 1], 0, 1)
print(f'v7b(정직 재학습) 단독 fold A BSS = {sc(pred_v7b_honest):.1f}')

# v88_raw 로드 (오늘 계속 쓴 것)
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

err_v7b = pred_v7b_honest - yv
err_v88 = v88_final - yv
rho = np.corrcoef(err_v7b, err_v88)[0, 1]
pred_rho = np.corrcoef(pred_v7b_honest, v88_final)[0, 1]
print(f'v88_final 단독 BSS = {sc(v88_final):.1f}')
print(f'corr(v7b_err, v88_err) = {rho:.4f}')
print(f'corr(v7b_pred, v88_pred) = {pred_rho:.4f}')
print()

B1 = np.mean(err_v88 ** 2)
B2 = np.mean(err_v7b ** 2)
thresh = np.sqrt(B1 / B2)
print(f'B1(v88 Brier)={B1:.6f}  B2(v7b Brier)={B2:.6f}  임계값 sqrt(B1/B2)={thresh:.6f}')
print(f'rho={rho:.6f}  {"통과(블렌드 이득 가능)" if rho < thresh else "미달(블렌드 불가)"}')

# 실제 최적 블렌드 가중치와 이득 계산 (C/V, 음수 허용)
resid_v88 = yv - v88_final
d = pred_v7b_honest - v88_final
C = np.mean(d * resid_v88)
V = np.mean(d ** 2)
a_star = C / V if V > 1e-12 else 0.0
blend = v88_final + a_star * d
print(f'최적 블렌드계수 a*={a_star:.4f}  블렌드 BSS={sc(blend):.1f}  (v88단독 대비 {sc(blend)-sc(v88_final):+.2f})')
