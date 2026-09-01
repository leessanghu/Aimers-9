"""진짜 '안 본 사람'(asof_pitcher_n=0, 첫 투구)을 따로 측정.
신규투수 그룹 전체는 이미 시즌 초반 경험을 쌓은 경우도 섞여있어서
콜드스타트(진짜 0경험) 효과와 뒤섞여 있었다. 경험량으로 쪼개서 분리한다.
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
pid = meta['pitcher_id'].to_numpy()
tr = season <= 2023; va = season == 2024
yv = y[va]; unc = 0.249807
seen_p = set(pid[tr])
pv = pid[va]
new_p = np.array([p not in seen_p for p in pv])

apn_log = X.loc[va, 'asof_pitcher_n'].to_numpy()
apn = np.expm1(apn_log).round().astype(int)
print('신규투수 그룹 내 asof_pitcher_n 분포:')
print(pd.Series(apn[new_p]).describe())
print()
print('정확히 n=0(첫 투구)인 행 수:', (apn[new_p] == 0).sum())
print('n=0~10:', ((apn[new_p] >= 0) & (apn[new_p] <= 10)).sum())
print('n=11~50:', ((apn[new_p] >= 11) & (apn[new_p] <= 50)).sum())
print('n=51+:', (apn[new_p] > 50).sum())

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
HARM = ['multires', 'midother', 'condball', 'countresid', 'future50', 'ingame']
W94 = {k: (v * 0.2 if k in HARM else v) for k, v in W.items()}
t = sum(W94.values()); W94 = {k: v / t for k, v in W94.items()}
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
p94 = sum(W94[k] * H[k] for k in W94)


def sc(m):
    return 1e5 * (1 - np.mean((np.clip(p94[m], 0, 1) - yv[m]) ** 2) / unc)


print()
print('=== 신규투수를 경험량으로 쪼개서 ===')
for lo, hi, nm in [(0, 0, 'n=0(첫투구)'), (1, 10, 'n=1~10'), (11, 50, 'n=11~50'), (51, 10**9, 'n=51+')]:
    m = new_p & (apn >= lo) & (apn <= hi)
    if m.sum() == 0:
        continue
    print(f'  {nm:14s} n={m.sum():>6,}  BSS={sc(m):8.1f}  실제={yv[m].mean():.4f} 예측={p94[m].mean():.4f} 편차={p94[m].mean()-yv[m].mean():+.4f} 예측std={p94[m].std():.4f}')

print()
print('=== 비교: 기존투수 중 저경험 구간 (경험량 효과 vs 신규여부 효과 분리) ===')
for lo, hi, nm in [(1, 10, 'n=1~10')]:
    m = (~new_p) & (apn >= lo) & (apn <= hi)
    print(f'  {nm:14s} n={m.sum():>6,}  BSS={sc(m):8.1f}  실제={yv[m].mean():.4f} 예측={p94[m].mean():.4f} 편차={p94[m].mean()-yv[m].mean():+.4f}')
