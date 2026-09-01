"""실험: LGBM + mc6pt 타겟(성공측을 구종축 fast/break/off로 재분할) 조합.
CatBoost판 mc6pt는 밤샘잡에서 학습중(몇시간 소요). LGBM은 훨씬 빨라서
지금 바로 신호 유무를 확인 가능 - 새 타겟축 + 다른 알고리즘 동시 테스트.

fold A만, v122 기준 직교화(vs d_mc6, d_xu) + 순열대조군 z. z>2면 실험적 프로브 후보.
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
ptype = np.load('dev/recovered_pitch_type.npy')

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
middle = valid & (mid > 0.5)
reverse = valid & (rev > 0.5) & (mid < 0.5)
nd = valid & (mid < 0.5) & (rev < 0.5)

ok_pt = ptype >= 0
is_fb, is_bk, is_os = ptype == 0, ptype == 1, ptype == 2
cls = np.full(n, -1, dtype=np.int64)
cls[middle] = 0
cls[reverse] = 1
cls[nd & (y == 0)] = 2
succ_ok = nd & (y == 1) & ok_pt
cls[succ_ok & is_fb] = 3
cls[succ_ok & is_bk] = 4
cls[succ_ok & is_os] = 5
SUCC = [3, 4, 5]
log('클래스 분포(mc6pt): ' + '  '.join(f'{c}:{(cls==c).mean()*100:.1f}%' for c in range(6))
    + f'  미분류:{(cls<0).mean()*100:.2f}%')

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


LGB = dict(objective='multiclass', num_class=6, n_estimators=800, learning_rate=0.05,
           num_leaves=63, min_child_samples=100, subsample=0.9, subsample_freq=1,
           colsample_bytree=0.7, reg_alpha=0.1, reg_lambda=1.0, random_state=42,
           n_jobs=-1, verbose=-1)

tr = (season <= upto) & (cls >= 0)
w = 0.5 ** ((upto - season[tr]) / 2.0)
Xtr, ctr = X.loc[tr], cls[tr]
n_es = int(len(Xtr) * 0.92)
ts = time.time()
m = LGBMClassifier(**LGB)
m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=w[:n_es],
      eval_set=[(Xtr.iloc[n_es:], ctr[n_es:])],
      callbacks=[early_stopping(50, verbose=False)])
proba = m.predict_proba(X.loc[va])
p = np.clip(proba[:, SUCC].sum(axis=1), 0, 1)
np.save('dev/cache_lgbmmc6pt_A.npy', p)
log(f'학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
print(f'단독 BSS = {sc(p):.2f}   (참고: mc6 CatBoost = {sc(p_mc6):.2f})')

d = p - blend; d -= d.mean()
V = float(np.mean(d ** 2)); A = float(np.mean(d * (blend - yv)))
rho0 = -A / np.sqrt(V * E_r2)
dp = orth(d, [d_mc6, d_xu])
Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
rho_p = -Ap / np.sqrt(Vp * E_r2) if Vp > 1e-16 else 0.0
print(f'원본:    rho={rho0:+.5f}  이득={K*A**2/V:+.2f}  s*={-A/V:+.4f}')
print(f'직교화후: rho={rho_p:+.5f}  이득={K*Ap**2/Vp if Vp>1e-16 else 0:+.2f}  '
      f's*={-Ap/Vp if Vp>1e-16 else 0:+.4f}')

ctrl = []
for sd in range(20):
    rng = np.random.RandomState(7000 + sd)
    dc = orth(rng.permutation(d), [d_mc6, d_xu])
    Vc = float(np.mean(dc ** 2))
    if Vc > 1e-16:
        ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
ctrl = np.array(ctrl)
z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
print(f'대조군: 평균={ctrl.mean():+.5f} SD={ctrl.std(ddof=1):.5f}')
print(f'z = {z:.1f}  ->  {"통과" if z>3 else ("경계" if z>1.5 else "허수")}')
log('완료')
