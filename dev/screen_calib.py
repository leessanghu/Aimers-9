"""캘리브레이션 효과 후보 스크리닝.
일반 피처가 아니라 '모델 출력에서 파생되는 메타 정보'를 본다.
  A) 예측값 p94 자체에 대한 재캘리브레이션 (구간별 잔차맵 = isotonic 유사)
  B) 헤드 간 불일치도 (std/range) — 162피처에 존재할 수 없는 정보
  C) mc5 확률분포의 불확실성 (entropy, maxprob)
전부 전역레벨 제거(평균중립) 후 순수 기여만 측정. fold A 내 H1<->H2 양방향.
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
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

# --- 후보 구성 ---
CORE = ['base', 'hurdle', 'ordinal', 'mc5']          # 살아남은 헤드들
Mcore = np.stack([H[k] for k in CORE], axis=1)
Mall = np.stack([H[k] for k in H], axis=1)
eps = 1e-9
P11c = np.clip(P11, eps, 1)

C = {}
C['p94_자체(재캘리브)'] = p94
C['헤드불일치_std(핵심4)'] = Mcore.std(axis=1)
C['헤드불일치_range(핵심4)'] = Mcore.max(axis=1) - Mcore.min(axis=1)
C['헤드불일치_std(전체10)'] = Mall.std(axis=1)
C['base_minus_hurdle'] = H['base'] - H['hurdle']
C['base_minus_mc5'] = H['base'] - H['mc5']
C['ordinal_minus_hurdle'] = H['ordinal'] - H['hurdle']
C['mc5_entropy'] = -(P11c * np.log(P11c)).sum(axis=1)
C['mc5_maxprob'] = P11.max(axis=1)
C['p94_dist_from_half'] = np.abs(p94 - 0.5)

print('=== 대조군: 순수 전역 레벨 보정 ===')
for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
    shift = resid[fit_m].mean()
    adj = p94.copy(); adj[ev_m] = p94[ev_m] + shift
    print(f'  {tag} shift={shift:+.5f} 이득={sc(adj, ev_m)-sc(p94, ev_m):+.2f}')
print()
print('=== 전역레벨 제거 후 순수 기여 (평균중립) ===')
print(f'{"후보":26s} {"H1->H2":>9s} {"H2->H1":>9s} {"평균":>8s} {"판정":>6s}')
print('-' * 64)
out = []
for name, v in C.items():
    v = np.asarray(v, dtype=np.float64)
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        edges = np.unique(np.quantile(v[fit_m], np.linspace(0, 1, 21)))
        if len(edges) < 3:
            gains.append(0.0); continue
        edges = edges.astype(float); edges[0] -= 1e-9; edges[-1] += 1e-9
        bf_ = np.clip(np.digitize(v[fit_m], edges) - 1, 0, len(edges) - 2)
        be = np.clip(np.digitize(v[ev_m], edges) - 1, 0, len(edges) - 2)
        rf = resid[fit_m]; gl = rf.mean()
        cmap = np.zeros(len(edges) - 1)
        for b in range(len(edges) - 1):
            m = bf_ == b
            if m.sum() >= 500:
                cmap[b] = rf[m].mean() - gl
        adj = p94.copy(); adj[ev_m] = p94[ev_m] + cmap[be]
        gains.append(sc(adj, ev_m) - sc(p94, ev_m))
    a = float(np.mean(gains)); vd = 'OK' if min(gains) > 0.3 else ''
    out.append((name, gains[0], gains[1], a, vd))
    print(f'{name:26s} {gains[0]:+9.2f} {gains[1]:+9.2f} {a:+8.2f} {vd:>6s}')

print()
ok = [r for r in out if r[4] == 'OK']
print(f'양방향 모두 +0.3 초과: {len(ok)}개')
for r in sorted(ok, key=lambda t: -t[3]):
    print(f'   {r[0]:26s} 평균 {r[3]:+.2f}')
