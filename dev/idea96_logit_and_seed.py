"""두 가지 측정:
(1) 로짓공간 블렌드 vs 확률공간 블렌드 (10헤드 전체, idea14는 3헤드만 했음)
(2) 시드앙상블 이득: 프로덕션은 aux head가 전부 seed=42 단일시드인데
    2시드 평균으로 바꾸면 얼마나 오르나 (GPU면 5~10시드도 가능)
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807
sc = lambda q: 1e5 * (1 - np.mean((np.clip(q, 0, 1) - yv) ** 2) / unc)

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
p = 'A'
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

# ---- 단일시드(프로덕션과 동일) vs 2시드 ----
def build(seedmode):
    """seedmode='s42' -> 프로덕션과 동일(단일시드), 'avg' -> 2시드평균"""
    def pick(base_path_fmt):
        if seedmode == 's42':
            return np.load(base_path_fmt.format(s=42))
        return np.mean([np.load(base_path_fmt.format(s=s)) for s in (42, 7)], axis=0)
    H = dict(
        base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=pick('dev/idea13_cache/' + p + '_multires_s{s}.npy'),
        ordinal=pick('dev/idea13_cache/' + p + '_ordinal_s{s}.npy'),
        midother=pick('dev/idea46_cache/' + p + '_midother_s{s}.npy'),
        condball=pick('dev/idea54_cache/' + p + '_cond_ball_s{s}.npy'),
        countresid=pick('dev/idea54_cache/' + p + '_count_resid_s{s}.npy'),
        future50=pick('dev/idea54_cache/' + p + '_future50_multi_s{s}.npy'),
    )
    P11 = np.load(f'dev/idea75_cache/{p}_proba11.npy')
    H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
    ing = np.load(f'dev/idea80_cache/{p}_ingame_heads_s42.npy')
    H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
    return H, P11

print('=== (2) 시드앙상블 이득 ===')
for mode, label in [('s42', '단일시드(프로덕션 현행)'), ('avg', '2시드평균')]:
    H, P11 = build(mode)
    raw = sum(W[k] * H[k] for k in H)
    risk = P11[:, [9, 10]].sum(axis=1)
    cut = np.maximum(0.0, risk - float(v88['risk_thr']))
    final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)
    print(f'  {label:24s} BSS={sc(final):.2f}')
    if mode == 's42':
        base_single = sc(final)
    else:
        print(f'  >>> 2시드 전환 이득 = {sc(final)-base_single:+.2f}')
print()

# ---- (1) 로짓 블렌드 ----
print('=== (1) 확률공간 vs 로짓공간 블렌드 (10헤드 전체) ===')
H, P11 = build('avg')
EPS = 1e-6
def logit(q):
    q = np.clip(q, EPS, 1 - EPS)
    return np.log(q / (1 - q))

prob_blend = sum(W[k] * H[k] for k in H)
logit_blend = 1 / (1 + np.exp(-sum(W[k] * logit(H[k]) for k in H)))

risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
def finalize(q):
    return np.clip(q - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

pf = finalize(prob_blend); lf = finalize(logit_blend)
print(f'  확률공간(현행)  BSS={sc(pf):.2f}   mean={pf.mean():.4f} std={pf.std():.4f}')
print(f'  로짓공간        BSS={sc(lf):.2f}   mean={lf.mean():.4f} std={lf.std():.4f}')
print(f'  >>> 로짓 전환 이득 = {sc(lf)-sc(pf):+.2f}')
print()

# 로짓블렌드 + 레벨 재보정 (로짓공간은 평균이 살짝 달라질 수 있어서)
H1 = mth <= 6; H2 = ~H1
resid_p = yv - pf
print('=== 로짓블렌드 H1<->H2 정직검증 (레벨 자동보정 포함) ===')
gains = []
for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
    shift = (yv[fit_m] - lf[fit_m]).mean()
    adj = lf.copy(); adj[ev_m] = lf[ev_m] + shift
    scm = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
    g = scm(adj, ev_m) - scm(pf, ev_m)
    gains.append(g)
    print(f'  {tag}: 로짓+레벨보정 vs 확률공간 = {g:+.2f}')
print(f'  평균 = {np.mean(gains):+.2f}')
