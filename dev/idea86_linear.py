"""선형모델(로지스틱 회귀) 테스트. 트리와 완전히 다른 귀납편향(축상분할 없음,
가산적 선형결합만) - MLP처럼 노이즈 과적합 위험 없이 낮은 상관을 낼 수 있는지 확인.
fold A(train<=2023 -> val 2024), 정직 검증. v88_raw와의 오차상관도 같이 본다."""
import numpy as np, pandas as pd, joblib, sys, time
sys.stdout.reconfigure(encoding='utf-8')
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
tr = season <= 2023; va = season == 2024
Xt, yt = X.loc[tr], y[tr]
Xv, yv = X.loc[va], y[va]
unc = 0.249807
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / unc)

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
v88_raw = sum(W[k] * H[k] for k in H)

log('스케일링...')
scaler = StandardScaler()
Xt_s = scaler.fit_transform(Xt.fillna(0.0))
Xv_s = scaler.transform(Xv.fillna(0.0))

print()
print(f'{"C(역정규화)":>12s} {"solo BSS":>10s} {"corr(err,v88err)":>18s} {"corr(pred,v88)":>15s}')
for Cval in [0.001, 0.01, 0.1, 1.0]:
    m = LogisticRegression(C=Cval, max_iter=200, n_jobs=-1)
    m.fit(Xt_s, yt)
    pred = m.predict_proba(Xv_s)[:, 1]
    solo = sc(pred)
    err_lin = pred - yv
    err_v88 = v88_raw - yv
    rho = np.corrcoef(err_lin, err_v88)[0, 1]
    rho_pred = np.corrcoef(pred, v88_raw)[0, 1]
    print(f'{Cval:12.3f} {solo:10.1f} {rho:18.4f} {rho_pred:15.4f}')
    log(f'  C={Cval} done')

print()
print('=== Ridge(선형회귀, 확률아닌 실수출력) 비교 ===')
for alpha in [1.0, 10.0, 100.0, 1000.0]:
    m = Ridge(alpha=alpha)
    m.fit(Xt_s, yt)
    pred = np.clip(m.predict(Xv_s), 0, 1)
    solo = sc(pred)
    rho = np.corrcoef(pred - yv, v88_raw - yv)[0, 1]
    rho_pred = np.corrcoef(pred, v88_raw)[0, 1]
    print(f'alpha={alpha:8.1f}  solo={solo:8.1f}  err_corr={rho:.4f}  pred_corr={rho_pred:.4f}')
