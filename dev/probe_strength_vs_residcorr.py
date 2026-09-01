"""'모델이 약해서 잔차상관이 낮다'는 가설 검증.
같은 162피처를 쓰는 '이미 강한' CatBoost 단독헤드(v95의 base)를 XGB/LGBM과 나란히 놓고
  (1) 단독 성능(fold 자체 BSS, 경쟁점수와 동일 스케일)
  (2) 나머지 7헤드 blend와의 잔차상관
을 비교한다. base가 훨씬 강한데도 잔차상관이 XGB/LGBM과 비슷하게 작다면
'강하게 키우면 해결된다'는 가설은 기각된다(오히려 강해질수록 같은 정보를 더 잘 짜내서
서로 수렴 = 상관 더 올라갈 위험).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def build_heads(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    H = build_heads(tag)
    # base 제외 7헤드 blend(=base 없이 나머지가 얼마나 강한지 + base의 잔차상관 기준점)
    W_all = {k: float(v95[f'{k}_weight']) for k in H}
    others = {k: v for k, v in W_all.items() if k != 'base'}
    t = sum(others.values()); Wo = {k: v / t for k, v in others.items()}
    blend_wo_base = np.clip(sum(Wo[k] * H[k] for k in others), 0, 1)
    resid_wo_base = yv - blend_wo_base

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B)
    base_bss = sc(H['base'])
    p_xgbrawid = np.load(f'dev/cache_xgbrawid_{tag}.npy')
    p_xgbctx = np.load(f'dev/cache_xgbctx_{tag}.npy')
    p_lgbm = np.load(f'dev/cache_lgbmrawid_{tag}.npy')
    xgb_bss = sc(p_xgbrawid)
    xgbctx_bss = sc(p_xgbctx)
    lgbm_bss = sc(p_lgbm)

    print(f'\n=== fold {tag} ===')
    print(f'  단독 BSS(경쟁점수 스케일):  base(CatBoost, 강함)={base_bss:8.1f}   '
          f'xgb_rawid={xgb_bss:8.1f}   xgb_ctx={xgbctx_bss:8.1f}   lgbm_rawid={lgbm_bss:8.1f}')

    d_base = H['base'] - blend_wo_base
    d_xgb = p_xgbrawid - blend_wo_base
    d_xgbctx = p_xgbctx - blend_wo_base
    d_lgbm = p_lgbm - blend_wo_base
    for name, d in [('base(CatBoost)', d_base), ('xgb_rawid', d_xgb),
                    ('xgb_ctx', d_xgbctx), ('lgbm_rawid', d_lgbm)]:
        cr = np.corrcoef(d, resid_wo_base)[0, 1]
        cp = np.corrcoef(d + blend_wo_base, blend_wo_base)[0, 1]
        print(f'    {name:<16} 예측corr(vs 나머지7blend)={cp:+.4f}   '
              f'잔차상관(vs 나머지7blend resid)={cr:+.4f}')

print('\n[해석] base(CatBoost)가 단독으로 XGB/LGBM보다 몇 배 강한데도 잔차상관이')
print(' 비슷하거나 더 크지 않다면 -> "모델을 강하게 키우면 해결"은 기각.')
print(' 오히려 강한 모델일수록 같은 정보를 더 잘 짜내 blend와 수렴(상관 상승) 위험이 있음.')
