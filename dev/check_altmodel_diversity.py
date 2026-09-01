"""이질 모델군(embMLP / TabM / LGBM / XGB)이 우리 GBDT 블렌드와 진짜로 다른가.
fold A(2024) 기준. 핵심지표는 '오차상관' - 오늘까지 본 옛 단일모델들은 전부 0.999+라
무의미했다. 0.99 미만이면 진짜 직교신호원 후보."""
import sys, glob, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

UNC = 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
va = season == 2024
yv = meta['control_success'].to_numpy(np.float64)[va]
base = np.load('dev/cache_v88_final_2024.npy')

df_raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['row_id', 'season'])
rid_2024 = df_raw.loc[df_raw['season'] == 2024, 'row_id'].to_numpy()
pos = {r: i for i, r in enumerate(rid_2024)}
print(f'2024 n={len(rid_2024):,}   v88_final n={len(base):,}')

sc = lambda p, m=None: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / UNC)
print(f'\nv88_final 단독 = {sc(base):.2f}\n')

resid_base = yv - base
X = pd.read_parquet('dev/featcache_X.parquet')
mth = X.loc[va, 'game_month'].to_numpy()
H1 = mth <= 6
H2 = ~H1
scm = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / UNC)

files = sorted(glob.glob('dev/phase3_preds/fold_2024_*.csv') + glob.glob('dev/phase4_preds/fold_2024_*.csv'))
print(f'{"모델":34s} {"단독BSS":>9s} {"오차상관":>9s} {"H1->H2":>8s} {"H2->H1":>8s} {"평균":>7s}')
for f in files:
    d = pd.read_csv(f)
    predcols = [c for c in d.columns if c not in ('row_id', 'y_valid')]
    idx = np.array([pos.get(r, -1) for r in d['row_id'].to_numpy()])
    ok = idx >= 0
    if ok.sum() < len(base) * 0.9:
        print(f'  {os.path.basename(f):32s} 매칭부족({ok.sum():,}/{len(base):,}) 스킵')
        continue
    for pc in predcols:
        p = np.full(len(base), np.nan)
        p[idx[ok]] = d[pc].to_numpy()[ok]
        if np.isnan(p).any():
            p = np.where(np.isnan(p), base, p)
        r = yv - p
        ecorr = np.corrcoef(resid_base, r)[0, 1]
        gains = []
        dd = p - base
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            C = np.mean(dd[fit_m] * resid_base[fit_m])
            V = np.mean(dd[fit_m] ** 2)
            wst = C / V if V > 1e-12 else 0.0
            bl = base.copy()
            bl[ev_m] = base[ev_m] + wst * dd[ev_m]
            gains.append(scm(bl, ev_m) - scm(base, ev_m))
        name = f'{os.path.basename(f).replace("fold_2024_","").replace(".csv","")}:{pc}'
        print(f'  {name:32s} {sc(p):>9.1f} {ecorr:>9.4f} {gains[0]:>+8.2f} {gains[1]:>+8.2f} {np.mean(gains):>+7.2f}')
