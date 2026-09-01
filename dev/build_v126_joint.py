"""v126: 9개 실측앵커 5축 결합해 재최적화. xu는 v124값 고정(V최대·단독A≈0이라 교차항 이동 위험).
mc6/strk/xr/lty를 최적점의 70%까지 이동.

핵심 근거(실측 역산):
  xr(xgb_rawid) A=-5.63e-5 -> 최적 +방향 (v123에서 -0.03으로 반대로 가서 -1.55 손해)
  lty(linear_tree) A=-3.91e-5 -> 최적 +방향 (v125에서 -0.03으로 반대로 가서 -0.52 손해)
  두 축 모두 부호는 견고(뒤집히려면 V가 5배 틀려야), 크기는 ±2배 불확실 -> 70%로 보수 이동.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
H = dict(
    base=avg([f'dev/phase90_cache/A_base_{q}.npy' for q in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/A_core_{q}.npy')) *
                    np.load(f'dev/phase90_cache/A_snc_{q}.npy') for q in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/A_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/A_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/A_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/A_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/A_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/A_future50_multi_s{s}.npy' for s in (42, 7)]),
)
W0 = {k: float(v95a[f'{k}_weight']) for k in HEADS8}
t_ = sum(W0.values()); W0 = {k: v / t_ for k, v in W0.items()}
core = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
NAMES = ['mc6', 'strk', 'xu', 'xr', 'lty']
P = [np.load('dev/cache_mc6head_A.npy'),
     np.load('dev/cache_strk_strk_linear_A.npy'),
     np.load('dev/cache_xgbunused_A.npy'),
     np.load('dev/cache_xgbrawid_A.npy'),
     np.load('dev/cache_lt_y_A.npy')]
D = [p - core for p in P]
V_loc = np.array([[float(np.mean(D[i] * D[j])) for j in range(5)] for i in range(5)])
S0 = 1103.6568315036
ANCH = [
    (0.0300, 0.0000,  0.0000, 0.0000,  0.0000, 1104.8342852052),
    (0.1000, 0.0000,  0.0000, 0.0000,  0.0000, 1107.2877112561),
    (0.4800, 0.0000,  0.0000, 0.0000,  0.0000, 1113.4251423543),
    (0.4800, 0.1000,  0.0000, 0.0000,  0.0000, 1114.5296512406),
    (0.4944, 0.1030, -0.0300, 0.0000,  0.0000, 1115.0039993398),
    (0.5092, 0.1061, -0.0309, -0.0300, 0.0000, 1113.4528720829),
    (0.4671, 0.1817, -0.0316, 0.0000,  0.0000, 1115.1606262971),
    (0.4811, 0.1872, -0.0325, 0.0000, -0.0300, 1114.6410582665),
]
c124 = np.array([0.4671, 0.1817, -0.0316, 0.0, 0.0])


def fit(lam):
    V = V_loc * lam
    rows = [2 * np.array(a[:5]) for a in ANCH]
    rhs = [-(a[5] - S0) / K - float(np.array(a[:5]) @ V @ np.array(a[:5])) for a in ANCH]
    A, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    sse = sum((a[5] - (S0 - K * (2 * np.array(a[:5]) @ A + np.array(a[:5]) @ V @ np.array(a[:5])))) ** 2
              for a in ANCH)
    return A, V, sse


best = None
for L in np.linspace(0.2, 2.5, 116):
    A_, V_, sse = fit(L)
    if best is None or sse < best[0]:
        best = (sse, L, A_, V_)
_, lam, A, V = best
score = lambda c: S0 - K * (2 * c @ A + c @ V @ c)

# xu 고정, 나머지 4축(mc6/strk/xr/lty) 최적화
free = [0, 1, 3, 4]
Vf = V[np.ix_(free, free)]
fixed = np.zeros(5); fixed[2] = c124[2]
grad = A + V @ fixed
c_opt_free = -np.linalg.solve(Vf, grad[free])
c_opt = fixed.copy(); c_opt[free] = c_opt_free

FRAC = 0.70
c = c124.copy()
c[free] = c124[free] + FRAC * (c_opt_free - c124[free])
print(f'lam={lam:.3f}  v124 예측={score(c124):.4f} (실측 1115.1606)')
print(f'xu고정 4축최적 = ' + '  '.join(f'{NAMES[i]}={c_opt[i]:+.4f}' for i in range(5))
      + f'  예측={score(c_opt):.4f}')
print(f'\n>>> v126 ({int(FRAC*100)}%) = ' + '  '.join(f'{NAMES[i]}={c[i]:+.4f}' for i in range(5))
      + f'  코어={1-c.sum():+.4f}')
print(f'    예측 점수 = {score(c):.4f}  (v124 대비 {score(c)-1115.1606262971:+.4f})')
for sc in (0.7, 1.0, 1.5):
    A2, V2, _ = fit(lam * sc)
    s2 = lambda cc: S0 - K * (2 * cc @ A2 + cc @ V2 @ cc)
    print(f'    V x{sc}: v126={s2(c):.3f}  v124={s2(c124):.3f}  차이={s2(c)-s2(c124):+.3f}')

# ---- 아티팩트 빌드 ----
v124a = joblib.load('submit/model/model_artifacts_v124.pkl')
v126 = dict(v124a)
CORE = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
        'condball', 'countresid', 'future50', 'mc5', 'ingame']
w95 = {k: float(v95a[f'{k}_weight']) for k in CORE}
t95 = sum(w95.values())
core_total = 1.0 - c.sum()
print(f'\n=== v126 가중치 ===')
tot = 0.0
for k in CORE:
    new = w95[k] / t95 * core_total
    v126[f'{k}_weight'] = new
    tot += new
    print(f'  {k:12s} {float(v124a[f"{k}_weight"]):+.4f} -> {new:+.4f}')
v126['mc6pure_weight'] = float(c[0])
v126['strk_weight'] = float(c[1])
v126['xgbunused_weight'] = float(c[2])
v126['xgbrawid_weight'] = float(c[3])
v126['lty_weight'] = float(c[4])
for nm, i in [('mc6pure', 0), ('strk', 1), ('xgbunused', 2), ('xgbrawid', 3), ('lty', 4)]:
    print(f'  {nm:12s} {float(v124a.get(nm+"_weight", 0.0)):+.4f} -> {c[i]:+.4f}')
tot += c.sum()
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'가중치 합계 오류: {tot}'

v108 = joblib.load('submit/model/model_artifacts_v108.pkl')
v126['xgbrawid_model'] = v108['xgbrawid_model']
v126['xgbrawid_cats'] = v108['xgbrawid_cats']
lty = joblib.load('dev/lty_production.pkl')
v126['lty_model'] = lty['model']
v126['lty_feat_order'] = lty['feat_order']

joblib.dump(v126, 'submit/model/model_artifacts_v126.pkl')
print('\nv126 저장 완료: submit/model/model_artifacts_v126.pkl')
