"""새 피처군(경기내 시퀀스)의 정직한 스크리닝.
질문: 이 피처들이 v94 블렌드가 못 잡는 잔차(y - p94)를 설명하는가?
방법: fold A(2024) 안에서 H1(1-6월)로 [피처 구간 -> 평균잔차] 맵을 만들고,
      H2(7-12월)에 적용해서 실제 BSS가 오르는지 본다. 반대방향도.
      모델 재학습이 없으므로 노이즈 0.
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
G = pd.read_parquet('dev/gameseq_feats.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
assert len(G) == len(X), (len(G), len(X))
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807

# v94 블렌드 재구성
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

FEATS = [c for c in G.columns if c not in ('season', 'control_success')]
Gv = G.loc[va].reset_index(drop=True)

print(f'v94 블렌드 fold A 전체 = {sc(p94, np.ones(len(yv), bool)):.1f}')
print(f'H1 n={H1.sum():,}  H2 n={H2.sum():,}')
print()
print(f'{"feature":18s} {"corr(y)":>9s} {"H1->H2":>9s} {"H2->H1":>9s} {"평균":>8s} {"판정":>6s}')
print('-' * 68)

rows = []
for c in FEATS:
    v = Gv[c].to_numpy(np.float64)
    r = np.corrcoef(v, yv)[0, 1]
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        # fit 구간에서 분위 구간별 평균잔차 맵 생성
        nb = min(20, len(np.unique(v)))
        if nb < 2:
            gains.append(0.0); continue
        edges = np.unique(np.quantile(v[fit_m], np.linspace(0, 1, nb + 1)))
        if len(edges) < 3:
            gains.append(0.0); continue
        edges[0] -= 1e-9; edges[-1] += 1e-9
        bi_fit = np.digitize(v[fit_m], edges) - 1
        bi_ev = np.clip(np.digitize(v[ev_m], edges) - 1, 0, len(edges) - 2)
        nbin = len(edges) - 1
        corr_map = np.zeros(nbin)
        rf = resid[fit_m]
        for b in range(nbin):
            m = bi_fit == b
            if m.sum() >= 200:
                corr_map[b] = rf[m].mean()
        adj = p94.copy()
        adj[ev_m] = p94[ev_m] + corr_map[bi_ev]
        gains.append(sc(adj, ev_m) - sc(p94, ev_m))
    avg_g = float(np.mean(gains))
    verdict = 'OK' if min(gains) > 0.3 else ''
    rows.append((c, r, gains[0], gains[1], avg_g, verdict))
    print(f'{c:18s} {r:+9.4f} {gains[0]:+9.2f} {gains[1]:+9.2f} {avg_g:+8.2f} {verdict:>6s}')

print()
ok = [r for r in rows if r[5] == 'OK']
print(f'양방향 모두 +0.3 초과: {len(ok)}개')
for r in sorted(ok, key=lambda t: -t[4]):
    print(f'   {r[0]:18s} 평균 {r[4]:+.2f}')
