"""중심화 버그 정정 후 모든 후보 재측정.
E[d*r] = Cov(d,r) + mean(d)*mean(r) 인데 fold A는 mean(resid)=-0.0072로 치우쳐 있어
중심화하지 않으면 레벨편차가 방향성 신호로 둔갑한다(v107을 승인시킨 오류).

올바른 자:  최대이득 = Cov(d,r)^2 / Var(d) * 400309
           +20 문턱: |Cov(d,r)|/sqrt(Var(d)) > 0.00707"""
import sys, glob, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

B = 0.249807
K = 1e5 / B
THRESH = 0.00707

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
va = season == 2024
yv = meta['control_success'].to_numpy(np.float64)[va]
blend = np.load('dev/cache_v88_final_2024.npy')
resid = yv - blend
print(f'fold A: mean(resid) = {resid.mean():+.6f}  <- 이 값이 비중심화 측정을 오염시킴')
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / B)


def measure(cand, tag):
    d = cand - blend
    md, mr = d.mean(), resid.mean()
    raw = np.mean(d * resid)          # 비중심화 (오염)
    cov = np.mean((d - md) * (resid - mr))
    var = np.mean((d - md) ** 2)
    if var <= 1e-14:
        return
    g_raw = (raw * raw / np.mean(d * d)) * K
    g_cen = (cov * cov / var) * K
    ratio = abs(cov) / np.sqrt(var)
    print(f'  {tag:34s} 단독={sc(cand):8.1f}  비중심화={g_raw:+8.2f}  '
          f'중심화={g_cen:+7.2f}  문턱대비={ratio/THRESH*100:5.1f}%')


print()
print('=== 물리헤드 (v107, 실측 -8.78) ===')
measure(np.load('dev/cache_physhead_2024.npy'), 'physhead')

print()
print('=== 이질 모델군 (fold_2024 예측) ===')
df_raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['row_id', 'season'])
rid = df_raw.loc[df_raw['season'] == 2024, 'row_id'].to_numpy()
pos = {r: i for i, r in enumerate(rid)}
for f in sorted(glob.glob('dev/phase3_preds/fold_2024_*.csv') + glob.glob('dev/phase4_preds/fold_2024_*.csv')):
    d_ = pd.read_csv(f)
    for pc in [c for c in d_.columns if c not in ('row_id', 'y_valid')]:
        idx = np.array([pos.get(r, -1) for r in d_['row_id'].to_numpy()])
        ok = idx >= 0
        if ok.sum() < len(blend) * 0.9:
            continue
        p = np.full(len(blend), np.nan)
        p[idx[ok]] = d_[pc].to_numpy()[ok]
        p = np.where(np.isnan(p), blend, p)
        measure(p, f'{os.path.basename(f)[10:-4]}:{pc}')

print()
print('=== 기존 10헤드 (참고: 이미 블렌드에 들어있음) ===')
import joblib
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
H = dict(
    base=avg([f'dev/phase90_cache/A_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/A_core_{n}.npy')) * np.load(f'dev/phase90_cache/A_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/A_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/A_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/A_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/A_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/A_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/A_future50_multi_s{s}.npy' for s in (42, 7)]),
)
P11 = np.load('dev/idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
for k in H:
    measure(H[k], k)
