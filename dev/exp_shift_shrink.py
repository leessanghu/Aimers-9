"""전역 레벨보정 크기를 축소해서 fold A/C 양쪽에서 안정적인 지점을 찾는다.
risk_alpha 때와 동일 원칙: 방향 고정(각 fit구간 자체 부호 사용), 크기(축소계수)만 스윕.
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
unc = 0.249807

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W8 = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
          ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
          countresid=v88['countresid_weight'], future50=v88['future50_weight'])
HARM = ['multires', 'midother', 'condball', 'countresid', 'future50']
W8s = {k: (v * 0.2 if k in HARM else v) for k, v in W8.items()}
t = sum(W8s.values()); W8s = {k: v / t for k, v in W8s.items()}
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def load8(p):
    return dict(
        base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{p}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{p}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


FOLD_DATA = {}
for p, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    mth = X.loc[va, 'game_month'].to_numpy()
    H = load8(p)
    pred = sum(W8s[k] * H[k] for k in W8s)
    resid = yv - pred
    FOLD_DATA[p] = (yv, pred, resid, mth <= 6, ~(mth <= 6))

sc = lambda pred, yv, m: 1e5 * (1 - np.mean((np.clip(pred[m], 0, 1) - yv[m]) ** 2) / unc)

print(f'{"lambda":>7s} {"A:H1->H2":>10s} {"A:H2->H1":>10s} {"C:H1->H2":>10s} {"C:H2->H1":>10s} {"최소":>8s} {"평균":>8s}')
for lam in [1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1, 0.05, 0.0]:
    row = []
    for p in ('A', 'C'):
        yv, pred, resid, H1, H2 = FOLD_DATA[p]
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            shift = resid[fit_m].mean() * lam
            adj = pred.copy(); adj[ev_m] = pred[ev_m] + shift
            g = sc(adj, yv, ev_m) - sc(pred, yv, ev_m)
            row.append(g)
    print(f'{lam:7.2f} {row[0]:10.2f} {row[1]:10.2f} {row[2]:10.2f} {row[3]:10.2f} {min(row):8.2f} {np.mean(row):8.2f}')
