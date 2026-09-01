"""g(x) 신호가 '진짜 없어서' 사라졌는지 'mc5_weight=0.138에 눌려서' 사라졌는지 분리.
1) g(x) 원신호(mc5 디코더 통과 전) 크기 확인
2) mc5_weight로 눌린 후 실제 얼마나 움직이는지
3) g(x) 유래 보정을 mc5 가중치와 독립적으로, 자기만의 최적계수 a*로 v88 위에 직접 추가
   (2K슬로프/risk_alpha와 동일한 C/V 방식). a*가 크게 나오면 '눌려서', ~0/음수면 '진짜없음'.
"""
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
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
raw = sum(W[k] * H[k] for k in H)
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
v88_final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

cls_tr = CLS5[tr]; y_tr = y[tr]
const_ball = y_tr[cls_tr == 2].mean()
p_ball_va = P11[:, [0, 1, 2]].sum(axis=1)  # P(nd&ball) 이 행이 ball일 확률

# g(x) 신호가 디코더를 통과했을 때 원래 만드는 raw 변화량 (mc5 자체 기준, 가중치 곱하기 전)
raw_signal_mc5 = p_ball_va * (g_pred_va - const_ball)   # mc5 예측값 자체의 변화분
print('=== 1) g(x) 신호 크기 (mc5 자체 값 기준, mc5_weight 곱하기 전) ===')
print(f'  mc5 예측 변화분 std = {raw_signal_mc5.std():.5f}')
print(f'  mc5_weight = {W["mc5"]:.4f}')
print(f'  v88_final에 실제 반영되는 변화분 std = {(W["mc5"]*raw_signal_mc5).std():.5f}')
print(f'  (참고: v88_final 자체 std = {v88_final.std():.5f})')
print()

# g(x) 신호를 mc5_weight와 무관하게, v88_final 위에 직접 additive로 붙여서
# 자기만의 최적계수(C/V)를 정직(H1/H2)으로 찾는다.
H1 = mth <= 6; H2m = ~H1
resid_base = yv - v88_final
signal = raw_signal_mc5  # 방향은 mc5 디코더와 동일한 신호(스케일만 재조정할 것)

print('=== 2) g(x) 신호를 독립 additive로: 자기만의 최적계수 a* (H1<->H2) ===')
gains = []
for fit_m, ev_m, tag in [(H1, H2m, 'H1->H2'), (H2m, H1, 'H2->H1')]:
    center = signal[fit_m].mean()
    cc = signal - center
    C = np.mean(cc[fit_m] * resid_base[fit_m])
    V = np.mean(cc[fit_m] ** 2)
    a = C / V if V > 1e-12 else 0.0
    adj = v88_final.copy(); adj[ev_m] = v88_final[ev_m] + a * cc[ev_m]
    g = sc(adj, ev_m) - sc(v88_final, ev_m)
    gains.append(g)
    print(f'  {tag}: fit에서 구한 a*={a:.4f}  (참고: mc5_weight=0.138이 곧 a=0.138과 동일한 스케일)  eval이득={g:+.2f}')
print(f'  평균 이득 = {np.mean(gains):+.2f}')
print()
print('  해석: a*가 0.138보다 훨씬 크면 -> 지금 mc5_weight에 눌려서 손해보는 중')
print('        a*가 0.138 근처거나 작으면 -> 이미 적정하게(혹은 과하게) 반영된 것')
