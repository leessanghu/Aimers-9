"""역산된 C(=E[(p-y)d])를 레벨항과 공분산항으로 분해.
C = Cov(p-y, d) + E[p-y]*E[d]
E[p-y] = D_true = -0.00097 (v106 프로브로 실측). E[d]는 fold로 추정.
레벨항이 C의 상당부분이면 -> XGB헤드를 레벨보정 후 넣으면 손해가 줄어든다.
레벨항이 무시가능하면 -> 판별력 자체가 잔차와 어긋나는 것이므로 튜닝으로 못 고침.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
D_TRUE = -0.00097
C_REAL = 2.718e-05

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


for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t = sum(W.values()); W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    p_xgb = np.load(f'dev/cache_xgbrawid_{tag}.npy')
    d = p_xgb - blend
    md = float(d.mean())
    lvl_term = D_TRUE * md
    print(f'fold{tag}: E[d]={md:+.5f}  레벨항(D_true*E[d])={lvl_term:+.3e}'
          f'  -> 실측C({C_REAL:.2e}) 대비 {100*lvl_term/C_REAL:+.2f}%')
    # 레벨중심화한 d로 다시 봤을 때 로컬 최대이득
    dc = d - md
    Cc = float(np.mean((blend - yv) * dc)); Vc = float(np.mean(dc ** 2))
    print(f'         레벨중심화 후 로컬: C={Cc:+.3e}  s*={-Cc/Vc:+.4f}  최대이득={K*Cc**2/Vc:+.2f}점')
