"""v94 블렌드 기준 risk 보정 재캘리브레이션.
fold A(2024)만 mc5(risk) 캐시가 있어서, 시간순 양방향 분할(H1<->H2)로 견고성 확인.
H1으로 thr/center/alpha 추정(train 역할) -> H2에서 실현 이득 측정(진짜 out-of-time).
반대방향(H2 추정 -> H1 검증)도 같이 봐서 한쪽 분할에만 의존 안 하게 함."""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
HARM = ['multires', 'midother', 'condball', 'countresid', 'future50', 'ingame']
W94 = {k: (v * 0.2 if k in HARM else v) for k, v in W.items()}
t = sum(W94.values())
W94 = {k: v / t for k, v in W94.items()}

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
P = np.load(f'dev/idea75_cache/{p}_proba11.npy')
H['mc5'] = np.clip(P @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load(f'dev/idea80_cache/{p}_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)

preds94 = sum(W94[k] * H[k] for k in W94)
risk_vec = P[:, [9, 10]].sum(axis=1)

sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
print(f'v94 블렌드(risk보정 전) fold A 전체 = {sc(preds94, np.ones(len(yv), bool)):.1f}')
print(f'  H1(1-6월) n={(mth<=6).sum():,}   H2(7-12월) n={(mth>6).sum():,}')
print()


def fit_thr_alpha(fit_mask, thr):
    cut = np.maximum(0.0, risk_vec - thr)
    center = cut[fit_mask].mean()
    cc = cut - center
    resid = yv - preds94
    C = np.mean(cc[fit_mask] * resid[fit_mask])
    V = np.mean(cc[fit_mask] ** 2)
    alpha_star = C / V if V > 1e-12 else 0.0
    return center, C, V, alpha_star


def eval_gain(mask_eval, thr, center, alpha):
    cut = np.maximum(0.0, risk_vec - thr)
    adj = preds94 - alpha * (cut - center)
    return sc(adj, mask_eval) - sc(preds94, mask_eval)


H1 = mth <= 6
H2 = ~H1
THRS = [0.15, 0.20, 0.25, 0.30, 0.35]

print(f'{"thr":>6s} {"fit->eval":>12s} {"center":>8s} {"C":>10s} {"V":>10s} {"alpha*":>8s} {"gain(full)":>11s} {"gain(0.5a)":>11s}')
for thr in THRS:
    for fit_mask, eval_mask, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
        center, C, V, a = fit_thr_alpha(fit_mask, thr)
        g_full = eval_gain(eval_mask, thr, center, a)
        g_half = eval_gain(eval_mask, thr, center, 0.5 * a)
        print(f'{thr:6.2f} {tag:>12s} {center:8.4f} {C:10.2e} {V:10.2e} {a:8.4f} {g_full:11.2f} {g_half:11.2f}')
    print()
