"""전역 레벨보정이 fold C에서 왜 손해인지 원인 진단.
H1/H2 각각의 실제 shift값 크기를 직접 비교."""
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


for p, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    mth = X.loc[va, 'game_month'].to_numpy()
    H = load8(p)
    pred = sum(W8s[k] * H[k] for k in W8s)
    resid = yv - pred
    H1 = mth <= 6; H2 = ~H1
    sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)

    print(f'=== fold {p} (val {vs}) ===')
    print(f'  월별 분포: H1(1-6월) n={H1.sum():,}  H2(7-12월) n={H2.sum():,}')
    print(f'  H1 평균잔차 = {resid[H1].mean():+.5f}   H2 평균잔차 = {resid[H2].mean():+.5f}')
    print(f'  전체 평균잔차 = {resid.mean():+.5f}')
    # 월별로 더 세밀하게
    print('  월별 평균잔차:')
    for m in sorted(np.unique(mth)):
        mm = mth == m
        print(f'    {int(m):2d}월 n={mm.sum():>6,}  잔차={resid[mm].mean():+.5f}  예측={pred[mm].mean():.4f}  실제={yv[mm].mean():.4f}')
    print()
    # H1 shift를 H2에 적용했을 때 실제로 뭐가 일어나는지
    shift_h1 = resid[H1].mean()
    adj = pred.copy(); adj[H2] = pred[H2] + shift_h1
    print(f'  H1shift({shift_h1:+.5f})를 H2에 적용: BSS {sc(pred,H2):.1f} -> {sc(adj,H2):.1f}  ({sc(adj,H2)-sc(pred,H2):+.2f})')
    shift_h2 = resid[H2].mean()
    adj2 = pred.copy(); adj2[H1] = pred[H1] + shift_h2
    print(f'  H2shift({shift_h2:+.5f})를 H1에 적용: BSS {sc(pred,H1):.1f} -> {sc(adj2,H1):.1f}  ({sc(adj2,H1)-sc(pred,H1):+.2f})')
    print()
