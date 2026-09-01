"""전체 상관이 아니라, 우리가 확인한 구체적 약점 구간에서 LightGBM이 우리 블렌드보다
나은지 확인. (1) 완전신인(asof_pitcher_n==0) (2) count_state==2(0-2카운트)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

B = 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
blend = np.load('dev/cache_v88_final_2024.npy')

raw = pd.read_csv('data/train.csv', encoding='utf-8-sig')
raw24 = raw[raw['season'] == 2024].reset_index(drop=True)
n_ = np.nan_to_num(raw24['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
count_state = (raw24['balls_before'] * 4 + raw24['strikes_before']).to_numpy()

lgbm_df = pd.read_csv('dev/phase4_preds/fold_2024_lgbm_variants.csv')
pos = {r: i for i, r in enumerate(raw24['row_id'].to_numpy())}
idx = np.array([pos.get(r, -1) for r in lgbm_df['row_id'].to_numpy()])
ok = idx >= 0


def eval_group(name, mask):
    yy = yv[mask]
    if yy.sum() == 0 or len(yy) < 20:
        print(f'  {name}: 표본부족(n={mask.sum()})')
        return
    r = yy.mean()
    var_own = max(r * (1 - r), 1e-6)
    pb = blend[mask]
    bs_b = np.mean((pb - yy) ** 2)
    print(f'\n  --- {name} (n={mask.sum():,}, 실제={r:.4f}) ---')
    print(f'    v88_final(우리)     예측={pb.mean():.4f}  편차={pb.mean()-r:+.5f}  자체BSS={1e5*(1-bs_b/var_own):8.1f}')
    for col in ['pred_A', 'pred_B', 'pred_C', 'pred_D']:
        p = np.full(len(blend), np.nan)
        p[idx[ok]] = lgbm_df[col].to_numpy()[ok]
        p = np.where(np.isnan(p), blend, p)
        pl = p[mask]
        bs_l = np.mean((pl - yy) ** 2)
        print(f'    lgbm:{col:8s}      예측={pl.mean():.4f}  편차={pl.mean()-r:+.5f}  자체BSS={1e5*(1-bs_l/var_own):8.1f}')


print('=== (1) 완전신인 (asof_pitcher_n == 0) ===')
eval_group('n==0', n_ < 0.5)
print('\n=== (2) 0-2 카운트 (count_state==2) ===')
eval_group('count_state==2', count_state == 2)
print('\n=== 참고: 전체 ===')
eval_group('전체', np.ones(len(yv), bool))
