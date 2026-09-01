"""외부 감사: 우리 10헤드 앙상블이 실제로 뭘 벌고 있는가.
(1) 각 헤드 단독 점수 vs 블렌드
(2) 주최측 제공 컬럼 1개만 쓴 자명한 모델
(3) 헤드간 예측 상관행렬 - 앙상블이라 부를 만한 다양성이 있는가"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
tr = season <= 2023
va = season == 2024
yv = y[va]
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / B)

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
keys = list(H.keys())
W = {k: float(v88[f'{k}_weight']) for k in keys}
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
blend_raw = sum(W[k] * H[k] for k in keys)
blend = np.clip(blend_raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center']))
                + float(v88['level_shift']), 0, 1)

print('=== (1) 각 헤드 단독 vs 블렌드 (fold A, 2024) ===')
solos = {}
for k in keys:
    s = sc(H[k])
    solos[k] = s
    print(f'  {k:12s} w={W[k]:.4f}  단독={s:8.2f}')
best_solo = max(solos.values())
best_name = max(solos, key=solos.get)
print(f'\n  최고 단독 헤드: {best_name} = {best_solo:.2f}')
print(f'  10헤드 블렌드(risk/level 보정 포함) = {sc(blend):.2f}')
print(f'  ★ 앙상블이 최고 단일헤드 대비 버는 값 = {sc(blend)-best_solo:+.2f}')
print(f'  단순평균(가중치 없이) = {sc(np.mean([H[k] for k in keys], axis=0)):.2f}')

print()
print('=== (2) 주최측 제공 컬럼만 쓴 자명한 모델 ===')
df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['season', 'asof_pitcher_success_rate', 'asof_pitcher_n'])
raw_rate = df.loc[va, 'asof_pitcher_success_rate'].to_numpy(np.float64)
gm = float(y[tr].mean())
print(f'  상수(train 평균 {gm:.4f})                = {sc(np.full(len(yv), gm)):8.2f}')
p_raw = np.where(np.isfinite(raw_rate), raw_rate, gm)
print(f'  asof_pitcher_success_rate 그대로       = {sc(p_raw):8.2f}')
n_ = df.loc[va, 'asof_pitcher_n'].to_numpy(np.float64)
n_ = np.nan_to_num(n_, nan=0.0)
for Kk in (200.0, 1000.0, 3000.0):
    p_sh = (n_ * np.nan_to_num(raw_rate, nan=gm) + Kk * gm) / (n_ + Kk)
    print(f'  위 값을 K={Kk:6.0f} 축소한 것            = {sc(p_sh):8.2f}')

print()
print('=== (3) 헤드간 예측 상관 (앙상블 다양성) ===')
M = np.corrcoef(np.array([H[k] for k in keys]))
print('      ' + ' '.join(f'{k[:6]:>7s}' for k in keys))
for i, k in enumerate(keys):
    print(f'{k[:6]:>6s} ' + ' '.join(f'{M[i,j]:7.3f}' for j in range(len(keys))))
off = M[np.triu_indices(len(keys), 1)]
print(f'\n  헤드간 상관 중앙값={np.median(off):.4f}  최소={off.min():.4f}  최대={off.max():.4f}')
