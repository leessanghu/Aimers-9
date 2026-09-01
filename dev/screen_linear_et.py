"""미검증 함수공간 2종 스크리닝: L2 로지스틱(순수선형) + ExtraTrees.

근거: xgbunused(+0.47 실측성공) 메커니즘 = 거친 모델을 빼서 블렌드의 편향 교정.
  선형모델은 궁극의 거친 모델(상호작용 0), ET는 부스팅과 다른 랜덤화의 배깅트리.
스크리닝: fold A, v126 기준, 직교화 8축(기존5+aux+nnraw+N1) + 순열대조군 z.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y_all = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95a['feature_order'])
X_raw = X_df[FEAT].astype(np.float64).to_numpy()

tr = season <= 2023
va = season == 2024
yv = y_all[va]
w_tr = 0.5 ** ((2023.0 - season[tr]) / 2.0)

mu = np.nanmean(X_raw[tr], axis=0)
sd = np.nanstd(X_raw[tr], axis=0) + 1e-9
Xz_tr = np.clip(np.nan_to_num((X_raw[tr] - mu) / sd, nan=0.0), -10, 10)
Xz_va = np.clip(np.nan_to_num((X_raw[va] - mu) / sd, nan=0.0), -10, 10)

# v126 블렌드 + 8축 직교화 준비
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50']
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
E_r2 = float(np.mean((yv - blend) ** 2))
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
BASES = [COMPS[k] - blend for k in ('mc6', 'strk', 'xu', 'xr', 'lty')]
BASES.append(np.load('dev/mc6family_cache/A_mc6aux.npy') - blend)
BASES.append(np.load('dev/cache_nnraw_A.npy') - blend)
BASES.append(np.load('dev/cache_nn_n1_A.npy') - blend)


def orth(dd, bases):
    dp = dd - dd.mean()
    for b in bases:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb < 1e-16:
            continue
        dp = dp - (float(np.mean(dp * b)) / Vb) * b
    return dp - dp.mean()


def screen(name, p):
    d = p - blend; d0 = d - d.mean()
    V = float(np.mean(d0 ** 2)); A = float(np.mean(d0 * (blend - yv)))
    dp = orth(d, BASES)
    Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
    rho_p = -Ap / np.sqrt(Vp * E_r2) if Vp > 1e-18 else 0.0
    ctrl = []
    for sd_ in range(20):
        rng = np.random.RandomState(16000 + sd_)
        dc = orth(rng.permutation(d0), BASES)
        Vc = float(np.mean(dc ** 2))
        if Vc > 1e-18:
            ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
    ctrl = np.array(ctrl)
    z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
    print(f'[{name:<8}] 단독BSS={sc(p):8.2f}  원본rho={-A/np.sqrt(V*E_r2):+.5f}  '
          f'직교후rho={rho_p:+.5f}  이득={K*Ap**2/Vp if Vp>1e-18 else 0:+.2f}  '
          f's*={-Ap/Vp if Vp>1e-18 else 0:+.4f}  z={z:5.1f}  {"통과" if z>2 else "허수"}', flush=True)


# 1) 순수 선형 (L2 로지스틱)
log('L2 로지스틱 학습...')
lr = LogisticRegression(C=1.0, max_iter=1000, solver='lbfgs', n_jobs=-1)
lr.fit(Xz_tr, y_all[tr], sample_weight=w_tr)
p_lin = np.clip(lr.predict_proba(Xz_va)[:, 1], 0, 1)
np.save('dev/cache_linear_A.npy', p_lin)
log('완료')
screen('linear', p_lin)

# 2) ExtraTrees
log('ExtraTrees 학습...')
et = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=200, max_features=0.4,
                          n_jobs=-1, random_state=42)
et.fit(X_df[FEAT].fillna(0).to_numpy()[tr], y_all[tr], sample_weight=w_tr)
p_et = np.clip(et.predict_proba(X_df[FEAT].fillna(0).to_numpy()[va])[:, 1], 0, 1)
np.save('dev/cache_et_A.npy', p_et)
log('완료')
screen('extratree', p_et)
log('전체 완료')
