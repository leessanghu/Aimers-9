"""TabM 정밀 검증 - 중심화 H1/H2 + 출처(정직성) 점검."""
import sys, os, glob
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

B = 0.249807
K = 1e5 / B
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
va = season == 2024
yv = meta['control_success'].to_numpy(np.float64)[va]
blend = np.load('dev/cache_v88_final_2024.npy')
resid = yv - blend
sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)
allm = np.ones(len(yv), bool)
mth = X.loc[va, 'game_month'].to_numpy()
H1 = mth <= 6
H2 = ~H1

df_raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['row_id', 'season'])
rid = df_raw.loc[df_raw['season'] == 2024, 'row_id'].to_numpy()
pos = {r: i for i, r in enumerate(rid)}


def load(f, col):
    d_ = pd.read_csv(f)
    idx = np.array([pos.get(r, -1) for r in d_['row_id'].to_numpy()])
    ok = idx >= 0
    p = np.full(len(blend), np.nan)
    p[idx[ok]] = d_[col].to_numpy()[ok]
    cov = ok.sum() / len(blend)
    return np.where(np.isnan(p), blend, p), cov


CANDS = [
    ('dev/phase3_preds/fold_2024_pred_tabm_pwl.csv', 'pred_tabm_pwl'),
    ('dev/phase3_preds/fold_2024_pred_tabm_base.csv', 'pred_tabm_base'),
    ('dev/phase4_preds/fold_2024_xgb_variants.csv', 'pred_xgb_l2_ids'),
    ('dev/phase4_preds/fold_2024_xgb_variants.csv', 'pred_xgb_log_ids'),
]

print('=== 중심화 H1/H2 정직검증 (fit구간에서 계수+절편 모두 추정 -> eval구간 적용) ===')
for f, col in CANDS:
    p, cov = load(f, col)
    d = p - blend
    gains = []
    coefs = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        md = d[fit_m].mean()
        mr = resid[fit_m].mean()
        covv = np.mean((d[fit_m] - md) * (resid[fit_m] - mr))
        varr = np.mean((d[fit_m] - md) ** 2)
        a = covv / varr if varr > 1e-14 else 0.0
        b = mr - a * md          # 절편도 fit구간에서 추정
        bl = blend.copy()
        bl[ev_m] = blend[ev_m] + a * d[ev_m] + b
        gains.append(sc(bl, ev_m) - sc(blend, ev_m))
        coefs.append(a)
    print(f'  {col:20s} cov={cov*100:.1f}%  a(H1)={coefs[0]:+.4f} a(H2)={coefs[1]:+.4f}  '
          f'H1->H2={gains[0]:+8.2f}  H2->H1={gains[1]:+8.2f}  평균={np.mean(gains):+8.2f}')

print()
print('=== 출처 점검: 이 예측이 정직한 OOF인가 ===')
for pat in ['dev/phase3*.py', 'dev/phase3_*.py', 'dev/*tabm*.py', 'dev/phase4*.py']:
    for g in glob.glob(pat):
        print(f'  발견: {g}')

print()
print('=== fold별 파일 존재 (2022/2023/2024 다 있으면 walk-forward로 만든 것) ===')
for g in sorted(glob.glob('dev/phase3_preds/*tabm_pwl*')):
    d_ = pd.read_csv(g)
    print(f'  {os.path.basename(g):40s} rows={len(d_):,}  pred평균={d_.iloc[:,-1].mean():.4f}')
