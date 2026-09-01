"""F/R 전용 레벨보정 검증.
train(fit) 구간에서만 그룹별 평균잔차를 계산해 상수로 더한다(Rule4 안전, v86 재발 방지).
H1<->H2 양방향 시간분할로 견고성 확인. 전역단일보정과 비교.
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
gt = meta['game_type'].to_numpy()
va = season == 2024
yv = y[va]; gtv = gt[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
HARM = ['multires', 'midother', 'condball', 'countresid', 'future50', 'ingame']
W94 = {k: (v * 0.2 if k in HARM else v) for k, v in W.items()}
t = sum(W94.values()); W94 = {k: v / t for k, v in W94.items()}
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
p94 = sum(W94[k] * H[k] for k in W94)

sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
resid = yv - p94
R = gtv == 'R'; F = gtv == 'F'
H1 = mth <= 6; H2 = ~H1

print(f'v94 무보정 전체 = {sc(p94, np.ones(len(yv), bool)):.1f}  (R={sc(p94,R):.1f}  F={sc(p94,F):.1f})')
print()

variants = {
    '전역 단일보정(대조군)': None,
    'R만 보정': 'R',
    'F만 보정': 'F',
    'R,F 둘다(각자 값)': 'RF',
}

for name, mode in variants.items():
    gains = []
    for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
        adj = p94.copy()
        if mode is None:
            shift = resid[fit_m].mean()
            adj[ev_m] = p94[ev_m] + shift
        elif mode == 'R':
            shift_r = resid[fit_m & R].mean()
            m = ev_m & R
            adj[m] = p94[m] + shift_r
        elif mode == 'F':
            shift_f = resid[fit_m & F].mean()
            m = ev_m & F
            adj[m] = p94[m] + shift_f
        elif mode == 'RF':
            shift_r = resid[fit_m & R].mean()
            shift_f = resid[fit_m & F].mean()
            mr = ev_m & R; mf = ev_m & F
            adj[mr] = p94[mr] + shift_r
            adj[mf] = p94[mf] + shift_f
        g_all = sc(adj, ev_m) - sc(p94, ev_m)
        g_R = sc(adj, ev_m & R) - sc(p94, ev_m & R)
        g_F = sc(adj, ev_m & F) - sc(p94, ev_m & F)
        gains.append((tag, g_all, g_R, g_F))
    print(f'=== {name} ===')
    for tag, g_all, g_R, g_F in gains:
        print(f'  {tag:8s} 전체={g_all:+7.2f}  R구간={g_R:+7.2f}  F구간={g_F:+7.2f}')
    avg_all = np.mean([g[1] for g in gains])
    print(f'  평균(전체) = {avg_all:+.2f}')
    print()
