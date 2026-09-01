"""각 헤드에 대해 blend 나머지를 고정하고 그 헤드의 최적계수(c*)를 H1/H2 양방향으로 재본다.
c*가 지금 가중치(W_old)보다 훨씬 작거나 음수면 '반대로 가야 할' 후보."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
unc = 0.249807
va = season == 2024
yv = y[va]

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
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

keys = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50', 'mc5', 'ingame']
W = {k: v88[f'{k}_weight'] for k in keys}
raw = sum(W[k] * H[k] for k in keys)
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
v88_final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

X_ = X.loc[va]
mth = X_['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
sc = lambda p_, m: 1e5 * (1 - np.mean((np.clip(p_[m], 0, 1) - yv[m]) ** 2) / unc)

print(f'{"head":12s} {"W_old":>7s} {"c*(H1->H2)":>11s} {"c*(H2->H1)":>11s} {"평균c*":>8s}  판정')
for k in keys:
    rest = v88_final - W[k] * H[k]  # 이 헤드를 뺀 나머지(risk보정 포함 그대로 유지, 근사)
    resid = yv - rest
    d = H[k]
    dc = d - d.mean()
    c_list = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        C = np.mean(dc[fit_m] * resid[fit_m])
        V = np.mean(dc[fit_m] ** 2)
        c_star = W[k] + (C / V if V > 1e-12 else 0.0)  # d의 중심화 계수이므로 W[k]에 더해줌(근사)
        c_list.append(c_star)
    avgc = np.mean(c_list)
    if avgc < 0:
        verdict = '*** 음수! 반대방향 후보 ***'
    elif avgc < W[k] * 0.3:
        verdict = '거의 0 - 빼도 될 후보'
    elif np.sign(c_list[0]) != np.sign(c_list[1]):
        verdict = '부호 불안정(둘 다 신뢰 어려움)'
    else:
        verdict = ''
    print(f'{k:12s} {W[k]:7.4f} {c_list[0]:11.4f} {c_list[1]:11.4f} {avgc:8.4f}  {verdict}')
log('완료')
