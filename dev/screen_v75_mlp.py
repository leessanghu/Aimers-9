"""v75 아카이브 MLP를 fold A에서 스크리닝 (유일한 비-트리 함수공간).

주의: v75는 2024 포함 전체데이터로 학습됨 -> fold A 평가는 in-sample 낙관편향.
따라서 이 스크린은 '한쪽 필터'로만 사용:
  - in-sample인데도 직교신호 ~0  -> 확실히 죽음 (프로브 불필요)
  - 신호 있어보임               -> 판단불가, 프로브(s=0.01)로만 결정 가능
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95a['feature_order'])
va = season == 2024
yv = y_all[va]

df = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['pitcher_id', 'batter_id'])
pid_all = df['pitcher_id'].to_numpy()
bid_all = df['batter_id'].to_numpy()

v75 = joblib.load('submit/model/model_artifacts_v75.pkl')
w = v75['mlp_weights']
FEAT75 = list(v75['feature_order'])
print(f'피처계약 일치: {FEAT75 == FEAT}')

Xv = X.loc[va, FEAT75].astype(np.float64)
pid = pid_all[va]; bid = bid_all[va]
ip = np.array([w['pmap'].get(v, 0) for v in pid], dtype=np.int64)
ib = np.array([w['bmap'].get(v, 0) for v in bid], dtype=np.int64)
unseen_p = np.mean([v not in w['pmap'] for v in pid])
print(f'fold A에서 pmap 미등록 투수비율: {unseen_p*100:.2f}%')

Xrow = Xv.to_numpy(np.float32)
z = np.clip((Xrow - w['mu']) / w['sd'], -10, 10)
h = np.concatenate([z, w['emb_p'][ip], w['emb_b'][ib]], axis=1)
h = np.maximum(h @ w['W1'] + w['b1'], 0)
h = np.maximum(h @ w['W2'] + w['b2'], 0)
logit = (h @ w['W3'] + w['b3']).squeeze(1)
p_mlp = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.0, 1.0).astype(np.float64)

# v126 블렌드
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
COMPS = dict(core=core,
             mc6=np.load('dev/cache_mc6head_A.npy'),
             strk=np.load('dev/cache_strk_strk_linear_A.npy'),
             xu=np.load('dev/cache_xgbunused_A.npy'),
             xr=np.load('dev/cache_xgbrawid_A.npy'),
             lty=np.load('dev/cache_lt_y_A.npy'))
W126 = dict(core=0.3491, mc6=0.4381, strk=0.1740, xu=-0.0316, xr=0.0354, lty=0.0350)
blend = np.clip(sum(W126[k] * COMPS[k] for k in COMPS), 0, 1)
resid = yv - blend
E_r2 = float(np.mean(resid ** 2))
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
print(f'MLP 단독 BSS(fold A, in-sample 낙관) = {sc(p_mlp):.2f}')
print(f'예측상관(MLP vs blend) = {np.corrcoef(p_mlp, blend)[0,1]:.4f}')

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


d = p_mlp - blend
d0 = d - d.mean()
V = float(np.mean(d0 ** 2)); A = float(np.mean(d0 * (blend - yv)))
print(f'원본:    rho={-A/np.sqrt(V*E_r2):+.5f}  V={V:.3e}  s*={-A/V:+.4f}')
dp = orth(d, BASES)
Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
rho_p = -Ap / np.sqrt(Vp * E_r2)
print(f'직교화후: rho={rho_p:+.5f}  이득={K*Ap**2/Vp:+.2f}  s*={-Ap/Vp:+.4f}')
ctrl = []
for sd in range(20):
    rng = np.random.RandomState(11000 + sd)
    dc = orth(rng.permutation(d0), BASES)
    Vc = float(np.mean(dc ** 2))
    if Vc > 1e-18:
        ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
ctrl = np.array(ctrl)
zc = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
print(f'대조군 z = {zc:.1f}')
print(f'\n해석: in-sample 낙관편향 있으므로 z 낮으면 확실히 죽음 / 높으면 판단불가(프로브 필요)')
np.save('dev/cache_v75mlp_A.npy', p_mlp)
