"""LGBM linear_tree: 리프 안에 선형회귀를 넣는 조각별-선형 모델.

근거: 우리 모델 전부(HGB/CatBoost/XGB/기본LGBM)가 계단함수(piecewise-constant).
     linear_tree는 함수공간 자체가 다름(piecewise-linear) - 축소평균 같은 매끄러운
     피처의 기울기 성분을 계단모델이 놓친다면 이게 잡는다.
     알고리즘 다양성(같은 함수공간)은 전부 죽었지만 함수공간 교체는 미검증.

두 타겟 모두 테스트:
  lt_y   : binary y 직접 (base와 비교되는 조각별선형 버전)
  lt_mc6 : mc6 6클래스 (가장 강한 축 + 새 함수공간)

검증: fold A, v122 기준, 직교화(vs d_mc6, d_xu) + 순열대조군 z. z>2만 후보.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from lightgbm import LGBMClassifier, early_stopping

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
order = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(n); lab[order] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
cls = np.full(n, -1, dtype=np.int64)
cls[valid & (mid > 0.5)] = 0
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y == 0)] = 2
cls[nd & (y == 1) & (ball > 0.5)] = 3
cls[nd & (y == 1) & (strike > 0.5)] = 4
cls[nd & (y == 1) & (inplay > 0.5)] = 5
SUCC = [3, 4, 5]

# v122 블렌드 (fold A)
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
S_MC6, S_STRK, S_XU = 0.48, 0.10, -0.03
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
W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
t_ = sum(W0.values()); W0 = {k: v / t_ for k, v in W0.items()}
blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
p_mc6 = np.load('dev/cache_mc6head_A.npy')
p_strk = np.load('dev/cache_strk_strk_linear_A.npy')
p_xu = np.load('dev/cache_xgbunused_A.npy')
rest = 1.0 - S_MC6 - S_STRK
blend = np.clip((rest * blend8 + S_MC6 * p_mc6 + S_STRK * p_strk) * (1 - S_XU) + S_XU * p_xu, 0, 1)

upto, vs = 2023, 2024
va = season == vs
yv = y[va]
resid = yv - blend
E_r2 = float(np.mean(resid ** 2))
SE = 1.0 / np.sqrt(len(yv))
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
d_mc6 = p_mc6 - blend
d_xu = p_xu - blend


def orth(dd, bases):
    dp = dd.copy()
    for b in bases:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb < 1e-16:
            continue
        dp = dp - (float(np.mean(dp * b)) / Vb) * b
    return dp - dp.mean()


def screen(name, p):
    d = p - blend; d -= d.mean()
    V = float(np.mean(d ** 2)); A = float(np.mean(d * (blend - yv)))
    rho0 = -A / np.sqrt(V * E_r2)
    dp = orth(d, [d_mc6, d_xu])
    Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
    rho_p = -Ap / np.sqrt(Vp * E_r2) if Vp > 1e-16 else 0.0
    ctrl = []
    for sd in range(20):
        rng = np.random.RandomState(6000 + sd)
        dc = orth(rng.permutation(d), [d_mc6, d_xu])
        Vc = float(np.mean(dc ** 2))
        if Vc > 1e-16:
            ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
    ctrl = np.array(ctrl)
    z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
    print(f'[{name}] 단독BSS={sc(p):8.2f}  rho={rho0:+.5f}  직교후rho={rho_p:+.5f}  '
          f'이득={K*Ap**2/Vp if Vp>1e-16 else 0:+6.2f}  s*={-Ap/Vp if Vp>1e-16 else 0:+.4f}  '
          f'z={z:5.1f}  {"통과" if z>3 else ("경계" if z>1.5 else "허수")}', flush=True)


LT_COMMON = dict(linear_tree=True, n_estimators=500, learning_rate=0.05,
                 num_leaves=31, min_child_samples=200, subsample=0.9, subsample_freq=1,
                 colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=5.0, random_state=42,
                 n_jobs=-1, verbose=-1)

tr_y = season <= upto
w_y = 0.5 ** ((upto - season[tr_y]) / 2.0)

# --- lt_y: binary y, 조각별선형 ---
log('lt_y 학습...')
Xtr = X.loc[tr_y]
n_es = int(len(Xtr) * 0.92)
m1 = LGBMClassifier(objective='binary', **LT_COMMON)
m1.fit(Xtr.iloc[:n_es], y[tr_y][:n_es], sample_weight=w_y[:n_es],
       eval_set=[(Xtr.iloc[n_es:], y[tr_y][n_es:])],
       callbacks=[early_stopping(50, verbose=False)])
p1 = np.clip(m1.predict_proba(X.loc[va])[:, 1], 0, 1)
np.save('dev/cache_lt_y_A.npy', p1)
log(f'lt_y 완료 best_iter={m1.best_iteration_}')
screen('lt_y  ', p1)

# --- lt_mc6: 6클래스, 조각별선형 ---
log('lt_mc6 학습...')
tr6 = (season <= upto) & (cls >= 0)
w6 = 0.5 ** ((upto - season[tr6]) / 2.0)
Xtr6, ctr6 = X.loc[tr6], cls[tr6]
n_es6 = int(len(Xtr6) * 0.92)
m2 = LGBMClassifier(objective='multiclass', num_class=6, **LT_COMMON)
m2.fit(Xtr6.iloc[:n_es6], ctr6[:n_es6], sample_weight=w6[:n_es6],
       eval_set=[(Xtr6.iloc[n_es6:], ctr6[n_es6:])],
       callbacks=[early_stopping(50, verbose=False)])
p2 = np.clip(m2.predict_proba(X.loc[va])[:, SUCC].sum(axis=1), 0, 1)
np.save('dev/cache_lt_mc6_A.npy', p2)
log(f'lt_mc6 완료 best_iter={m2.best_iteration_}')
screen('lt_mc6', p2)
log('전체 완료')
