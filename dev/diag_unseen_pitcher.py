"""미지의 투수/타자 진단.
test는 2025 시즌 245,789행. train(2019-2024)에 없던 투수가 상당수 나올 수 있다.
fold A가 같은 구조(train<=2023, val=2024)이므로 여기서 직접 측정 가능."""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
pid = meta['pitcher_id'].to_numpy(); bid = meta['batter_id'].to_numpy()
tr = season <= 2023; va = season == 2024
yv = y[va]; unc = 0.249807

seen_p = set(pid[tr]); seen_b = set(bid[tr])
pv = pid[va]; bv = bid[va]
new_p = np.array([p not in seen_p for p in pv])
new_b = np.array([b not in seen_b for b in bv])

print(f'val(2024) 총 {va.sum():,}행')
print(f'  train에 없던 투수의 행: {new_p.sum():,} ({new_p.mean()*100:.2f}%)  고유투수={len(set(pv[new_p]))}')
print(f'  train에 없던 타자의 행: {new_b.sum():,} ({new_b.mean()*100:.2f}%)  고유타자={len(set(bv[new_b]))}')
print(f'  둘 다 처음: {(new_p & new_b).sum():,}')
print()

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


allm = np.ones(len(yv), bool)
print('=== 그룹별 BSS (v94 블렌드, fold A) ===')
print(f'  전체         {sc(allm):8.1f}   n={allm.sum():,}')
print(f'  기존 투수    {sc(~new_p):8.1f}   n={(~new_p).sum():,}')
print(f'  신규 투수    {sc(new_p):8.1f}   n={new_p.sum():,}')
print()
print(f'  기존 타자    {sc(~new_b):8.1f}   n={(~new_b).sum():,}')
print(f'  신규 타자    {sc(new_b):8.1f}   n={new_b.sum():,}')
print()
print('=== 예측 vs 실제 (레벨/분산) ===')
for nm, m in [('신규투수', new_p), ('기존투수', ~new_p)]:
    print(f'  {nm}: 실제={yv[m].mean():.4f} 예측={p94[m].mean():.4f} 편차={p94[m].mean()-yv[m].mean():+.4f} 예측std={p94[m].std():.4f}')
print()
# 투수 경험량(asof_pitcher_n)별로도 쪼개보기
X = pd.read_parquet('dev/featcache_X.parquet')
apn = X.loc[va, 'asof_pitcher_n'].to_numpy()   # log1p 되어있음
q = pd.qcut(apn, 5, labels=False, duplicates='drop')
print('=== asof_pitcher_n(경험량) 5분위별 ===')
for b in range(int(q.max()) + 1):
    m = q == b
    print(f'  q{b}: n={m.sum():>7,}  BSS={sc(m):8.1f}  실제={yv[m].mean():.4f} 예측={p94[m].mean():.4f} 편차={p94[m].mean()-yv[m].mean():+.4f}')
