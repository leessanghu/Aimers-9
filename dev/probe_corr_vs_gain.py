"""'피처를 다르게 가면 예측상관이 내려가고 그게 이득으로 이어지나'를 직접 검증.
xgb_rawid(162+ID) vs xgb_ctx(51+ID, 축소평균 제거) vs lgbm_rawid(162+ID) 3종에 대해
  (1) 블렌드와의 예측상관(corr)
  (2) 클린 max-gain(이미 측정됨, fold A)
을 나란히 놓고 상관 하락이 실제로 이득 증가로 이어지는지, 어디서 멈추는지 본다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def build8(tag):
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


va = season == 2024
yv = y[va]
H = build8('A')
W = {k: float(v95[f'{k}_weight']) for k in H}
t = sum(W.values()); W = {k: v / t for k, v in W.items()}
blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
resid = yv - blend

CANDS = {
    'xgb_rawid(162+ID)': ('dev/cache_xgbrawid_A.npy', 0.90),
    'xgb_ctx(51+ID)   ': ('dev/cache_xgbctx_A.npy', 1.49),
    'lgbm_rawid(162+ID)': ('dev/cache_lgbmrawid_A.npy', 0.23),
}
print(f'{"후보":<20}{"예측corr(vs blend)":>20}{"d=p-blend std":>16}{"resid와 corr":>14}{"클린maxgain":>12}')
for name, (path, gain) in CANDS.items():
    p = np.load(path)
    d = p - blend
    corr_pred = np.corrcoef(p, blend)[0, 1]
    corr_resid = np.corrcoef(d, resid)[0, 1]
    print(f'{name:<20}{corr_pred:>20.4f}{d.std():>16.4f}{corr_resid:>14.4f}{gain:>+11.2f}점')

print('\n대조군(랜덤노이즈, blend와 상관 0으로 설계): 예측corr~0, 잔차corr~0, 클린maxgain=+1.69'
      '\n-> 예측상관이 완전히 0인 "다름"도 그 자체로는 최대 +1.69점(대조군) 이상을 보장 못함'
      '\n-> 이득은 잔차상관(corr_resid) 크기가 결정. 예측상관 하락은 필요조건이지 충분조건이 아님')
