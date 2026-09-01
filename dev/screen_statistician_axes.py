"""통계학자 관점 무학습 후보 3종을 v126 기준으로 스크리닝.

1. logit_pool : 같은 가중치로 로짓공간 풀링 - 선형풀 대비 적응적 선명화.
   (Ranjan-Gneiting: 상관 높은 예측자들의 선형풀은 과소확신 -> 로짓풀이 교정)
2. winsor     : 꼬리 수축 방향(상하위 5% 예측만 분위수로 클립) - 꼬리 캘리브만 분리.
3. agree_sharp: 헤드 불일치(spread)와 선명화의 교호작용 - 합의 강한 곳만 선명화.

검증: fold A, v126 기준, 5개 기해결축(mc6/strk/xu/xr/lty) 전부에 직교화 + 순열대조군 z.
전부 무학습이라 즉시 실행. z>2만 프로브 후보.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
va = season == 2024
yv = y_all[va]
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
COMPS = dict(
    core=core,
    mc6=np.load('dev/cache_mc6head_A.npy'),
    strk=np.load('dev/cache_strk_strk_linear_A.npy'),
    xu=np.load('dev/cache_xgbunused_A.npy'),
    xr=np.load('dev/cache_xgbrawid_A.npy'),
    lty=np.load('dev/cache_lt_y_A.npy'),
)
W126 = dict(core=0.3491, mc6=0.4381, strk=0.1740, xu=-0.0316, xr=0.0354, lty=0.0350)
blend = np.clip(sum(W126[k] * COMPS[k] for k in COMPS), 0, 1)
resid = yv - blend
E_r2 = float(np.mean(resid ** 2))
n = len(yv)
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
print(f'v126 fold A BSS = {sc(blend):.2f}')

# 기해결 5축 방향 (v126 기준)
BASES = [COMPS[k] - blend for k in ('mc6', 'strk', 'xu', 'xr', 'lty')]


def orth(dd, bases):
    dp = dd - dd.mean()
    for b in bases:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb < 1e-16:
            continue
        dp = dp - (float(np.mean(dp * b)) / Vb) * b
    return dp - dp.mean()


def screen(name, p_alt):
    d = p_alt - blend
    d0 = d - d.mean()
    V = float(np.mean(d0 ** 2))
    A = float(np.mean(d0 * (blend - yv)))
    rho0 = -A / np.sqrt(V * E_r2)
    dp = orth(d, BASES)
    Vp = float(np.mean(dp ** 2))
    if Vp < 1e-18:
        print(f'[{name}] 기존축과 완전중복')
        return
    Ap = float(np.mean(dp * (blend - yv)))
    rho_p = -Ap / np.sqrt(Vp * E_r2)
    ctrl = []
    for sd in range(20):
        rng = np.random.RandomState(9000 + sd)
        dc = orth(rng.permutation(d0), BASES)
        Vc = float(np.mean(dc ** 2))
        if Vc > 1e-18:
            ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
    ctrl = np.array(ctrl)
    z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
    print(f'[{name:<12}] rho={rho0:+.5f}  직교후rho={rho_p:+.5f}  직교후이득={K*Ap**2/Vp:+.2f}  '
          f's*={-Ap/Vp:+.4f}  z={z:5.1f}  {"통과" if z>3 else ("경계" if z>1.5 else "허수")}')


EPS = 1e-6
lg = lambda p: np.log(np.clip(p, EPS, 1-EPS) / (1 - np.clip(p, EPS, 1-EPS)))
sig = lambda x: 1 / (1 + np.exp(-x))

# 1. 로짓풀 (같은 가중치, 로짓공간)
z_lp = sum(W126[k] * lg(COMPS[k]) for k in COMPS)
p_logit = sig(z_lp)
print(f'\n로짓풀 통계: 선형풀SD={blend.std():.5f} -> 로짓풀SD={p_logit.std():.5f} '
      f'(로짓풀이 더 선명해야 정상)')
screen('logit_pool', p_logit)

# 2. 꼬리 수축 (상하위 5%만 분위수로 클립)
q05, q95 = np.quantile(blend, 0.05), np.quantile(blend, 0.95)
p_wins = np.clip(blend, q05, q95)
screen('winsor', p_wins)

# 3. 합의조건부 선명화: spread 낮을수록 선명화 크게
mat = np.column_stack([COMPS[k] for k in ('core', 'mc6', 'strk')])
spread = mat.std(axis=1)
w_sh = 1.0 / (1.0 + spread / spread.mean())
p_agree = blend + 0.10 * w_sh * (blend - blend.mean())
screen('agree_sharp', p_agree)
