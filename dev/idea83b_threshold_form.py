"""idea83 재시도: 20분위 구간화 대신 원래 threshold 형태(max(0,risk-thr))로.
mc5_risk의 known-good(+4.54/+1.33)을 재현 확인 후, ordinal_risk/불일치도도 같은 형태로."""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807

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
v88_raw = sum(W[k] * H[k] for k in W)

sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
resid_base = yv - v88_raw
H1 = mth <= 6; H2 = ~H1

mc5_risk = P11[:, [9, 10]].sum(axis=1)
ordinal_risk = 1.0 - H['ordinal']
mz = (mc5_risk - mc5_risk.mean()) / mc5_risk.std()
oz = (ordinal_risk - ordinal_risk.mean()) / ordinal_risk.std()
disagree = np.abs(mz - oz)


def threshold_scan(axis, thrs, alpha_fixed=None):
    """각 thr에서 fit구간 C/V로 alpha* 구하고 eval에 적용. alpha_fixed 주면 그 값 고정 사용."""
    out = []
    for thr in thrs:
        cut = np.maximum(0.0, axis - thr)
        gains = []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            center = cut[fit_m].mean()
            cc = cut - center
            if alpha_fixed is None:
                C = np.mean(cc[fit_m] * resid_base[fit_m])
                V = np.mean(cc[fit_m] ** 2)
                a = C / V if V > 1e-12 else 0.0
            else:
                a = alpha_fixed
            adj = v88_raw.copy(); adj[ev_m] = v88_raw[ev_m] - a * cc[ev_m]
            gains.append(sc(adj, ev_m) - sc(v88_raw, ev_m))
        out.append((thr, gains[0], gains[1], np.mean(gains)))
    return out


print('=== mc5_risk (sanity check, thr=0.25 alpha=0.045 고정으로 재현) ===')
for thr, g1, g2, ga in threshold_scan(mc5_risk, [0.25], alpha_fixed=0.045):
    print(f'  thr={thr:.2f}  H1->H2={g1:+.2f}  H2->H1={g2:+.2f}  평균={ga:+.2f}')
print()

print('=== ordinal_risk: threshold 스캔 (분위수 기반) ===')
qs = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
thrs = [np.quantile(ordinal_risk, q) for q in qs]
for (thr, g1, g2, ga), q in zip(threshold_scan(ordinal_risk, thrs), qs):
    print(f'  q={q:.2f}(thr={thr:.4f})  H1->H2={g1:+7.2f}  H2->H1={g2:+7.2f}  평균={ga:+7.2f}')
print()

print('=== disagree(불일치도): threshold 스캔 ===')
qs2 = [0.5, 0.7, 0.8, 0.9, 0.95]
thrs2 = [np.quantile(disagree, q) for q in qs2]
for (thr, g1, g2, ga), q in zip(threshold_scan(disagree, thrs2), qs2):
    print(f'  q={q:.2f}(thr={thr:.4f})  H1->H2={g1:+7.2f}  H2->H1={g2:+7.2f}  평균={ga:+7.2f}')
print()

print('=== mc5_risk + ordinal_risk 결합 (둘다 위험할 때만, min(mz,oz)) ===')
comb = np.minimum(mz, oz)
qs3 = [0.5, 0.7, 0.8, 0.9]
thrs3 = [np.quantile(comb, q) for q in qs3]
for (thr, g1, g2, ga), q in zip(threshold_scan(comb, thrs3), qs3):
    print(f'  q={q:.2f}(thr={thr:.4f})  H1->H2={g1:+7.2f}  H2->H1={g2:+7.2f}  평균={ga:+7.2f}')
