"""스크리닝 v2: 전역 레벨 보정을 대조군으로 두고, 피처의 '순수' 기여만 분리.
1차 스크리닝(screen_gameseq.py)은 구간 평균잔차를 그대로 더해서, 값이 한쪽에 몰린
변수에서는 사실상 전역 레벨 보정이 섞여 들어갔다. 여기서는 구간효과에서 전역평균을
빼서 평균중립(centered) 보정만 평가한다."""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
G = pd.read_parquet('dev/gameseq_feats.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024; yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy(); unc = 0.249807

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
P11 = np.load(f'dev/idea75_cache/{p}_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load(f'dev/idea80_cache/{p}_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
p94 = sum(W94[k] * H[k] for k in W94)

sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
H1 = mth <= 6; H2 = ~H1
resid = yv - p94

print('=== 대조군: 순수 전역 레벨 보정만 (피처 정보 0) ===')
for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
    shift = resid[fit_m].mean()
    adj = p94.copy(); adj[ev_m] = p94[ev_m] + shift
    print(f'  {tag}  레벨shift={shift:+.5f}  이득={sc(adj, ev_m)-sc(p94, ev_m):+.2f}')
print(f'  mean resid H1={resid[H1].mean():+.5f}  H2={resid[H2].mean():+.5f}')
print()

print('=== 전역레벨 제거 후 = 피처의 순수 기여 (평균중립 보정) ===')
Gv = G.loc[va].reset_index(drop=True)
FEATS = [c for c in G.columns if c not in ('season', 'control_success')]
print(f'{"feature":18s} {"H1->H2":>9s} {"H2->H1":>9s} {"평균":>8s} {"판정":>6s}')
print('-' * 56)
res = []
for c in FEATS:
    v = Gv[c].to_numpy(np.float64)
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        uniq = np.unique(v)
        if len(uniq) < 2:
            gains.append(0.0); continue
        if len(uniq) <= 20:
            edges = np.r_[uniq - 1e-9, uniq[-1] + 1e-9]
        else:
            edges = np.unique(np.quantile(v[fit_m], np.linspace(0, 1, 21)))
            if len(edges) < 3:
                gains.append(0.0); continue
            edges = edges.astype(float); edges[0] -= 1e-9; edges[-1] += 1e-9
        bf_ = np.clip(np.digitize(v[fit_m], edges) - 1, 0, len(edges) - 2)
        be = np.clip(np.digitize(v[ev_m], edges) - 1, 0, len(edges) - 2)
        nbin = len(edges) - 1
        rf = resid[fit_m]; gl = rf.mean()
        cmap = np.zeros(nbin)
        for b in range(nbin):
            m = bf_ == b
            if m.sum() >= 500:
                cmap[b] = rf[m].mean() - gl      # 전역레벨 제거 = 평균중립
        adj = p94.copy(); adj[ev_m] = p94[ev_m] + cmap[be]
        gains.append(sc(adj, ev_m) - sc(p94, ev_m))
    a = float(np.mean(gains))
    vd = 'OK' if min(gains) > 0.3 else ''
    res.append((c, gains[0], gains[1], a, vd))
    print(f'{c:18s} {gains[0]:+9.2f} {gains[1]:+9.2f} {a:+8.2f} {vd:>6s}')

print()
ok = [r for r in res if r[4] == 'OK']
print(f'양방향 모두 +0.3 초과: {len(ok)}개')
for r in sorted(ok, key=lambda t: -t[3]):
    print(f'   {r[0]:18s} 평균 {r[3]:+.2f}')
