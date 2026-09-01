"""mc6 R/F 전문가 분리(mixture-of-experts): R행은 R전문가, F행은 F전문가로 라우팅.

가설: F리그는 판정 레짐이 다름(2022->2023 성공률 0.71->0.47 대단절). 공유트리가
  두 레짐을 한 모델에 섞으면서 손해보고 있다면, 리그별 전문가 분리가 이득.
R-only 단독이 아니라 혼합 라우팅인 이유: test 5행의 all-R은 '형식확인용 표본'일 뿐
  (공식 문서 명시) - 2025 구성 가정 없이 game_type(추론시 가용)으로 라우팅하면
  테스트 구성이 뭐든 성립.
fold A에서 합성예측 스크리닝(기존 mc6 직교화 + 대조군 z) -> 통과시 프로덕션 2모델.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y_all = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95a['feature_order'])
X = X_df[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
test_gt = pd.read_csv('data/test.csv', encoding='utf-8-sig', usecols=['game_type'])
r_value = test_gt['game_type'].iloc[0]
is_R = (df['game_type'] == r_value).to_numpy()
print(f'test game_type 값 = {r_value!r}')
print(f'train에서 R 행 비율 = {is_R.mean()*100:.1f}%  (R 성공률={y_all[is_R].mean():.4f}, '
      f'비R 성공률={y_all[~is_R].mean():.4f})')

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


rev = diff_label('asof_pitcher_reverse_rate'); mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
cls = np.full(n, -1, dtype=np.int64)
cls[valid & (mid > 0.5)] = 0
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y_all == 0)] = 2
cls[nd & (y_all == 1) & (ball > 0.5)] = 3
cls[nd & (y_all == 1) & (strike > 0.5)] = 4
cls[nd & (y_all == 1) & (inplay > 0.5)] = 5
SUCC = [3, 4, 5]

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=6, early_stopping_rounds=50,
          random_seed=42)

# ---- fold A: R전문가 + F전문가 학습, game_type 라우팅 합성 ----
va = season == 2024
yv = y_all[va]


def train_expert(mask_league, name):
    tr = (season <= 2023) & (cls >= 0) & mask_league
    w = 0.5 ** ((2023.0 - season[tr]) / 2.0)
    Xtr, ctr = X.loc[tr], cls[tr]
    n_es = int(len(Xtr) * 0.92)
    log(f'{name} 학습행 {tr.sum():,}')
    m = CatBoostClassifier(**CB)
    m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=w[:n_es],
          eval_set=(Xtr.iloc[n_es:], ctr[n_es:]))
    proba = m.predict_proba(X.loc[va])
    p = np.clip(proba[:, SUCC].sum(axis=1), 0, 1)
    log(f'{name} 완료 best_iter={m.best_iteration_}')
    return p


p_R = train_expert(is_R, 'R전문가')
p_F = train_expert(~is_R, 'F전문가')
gt_va = is_R[va]
p_r = np.where(gt_va, p_R, p_F)   # 라우팅 합성
np.save('dev/cache_mc6split_A.npy', p_r)
log('라우팅 합성 완료')

# ---- 스크리닝 ----
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
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
BASES = [COMPS[k] - blend for k in ('mc6', 'strk', 'xu', 'xr', 'lty')]


def orth(dd, bases):
    dp = dd - dd.mean()
    for b in bases:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb > 1e-16:
            dp = dp - (float(np.mean(dp * b)) / Vb) * b
    return dp - dp.mean()


# R행/F행 나눠서 평가 (전문가가 각자 구역에서 공유트리를 이기는지가 핵심)
scm = lambda pp, m: 1e5 * (1 - np.mean((np.clip(pp[m], 0, 1) - yv[m]) ** 2) / B_)
print(f'\nmc6_split(라우팅) 단독 BSS = {sc(p_r):.2f}  (기존 mc6 = {sc(COMPS["mc6"]):.2f})')
print(f'  R행({gt_va.sum():,}):  전문가={scm(p_r, gt_va):.2f}  공유mc6={scm(COMPS["mc6"], gt_va):.2f}')
print(f'  F행({(~gt_va).sum():,}): 전문가={scm(p_r, ~gt_va):.2f}  공유mc6={scm(COMPS["mc6"], ~gt_va):.2f}')
print(f'기존 mc6와 예측상관 = {np.corrcoef(p_r, COMPS["mc6"])[0,1]:.4f}')

d = p_r - blend; d0 = d - d.mean()
V = float(np.mean(d0 ** 2)); A = float(np.mean(d0 * (blend - yv)))
print(f'원본:    rho={-A/np.sqrt(V*E_r2):+.5f}  s*={-A/V:+.4f}')
dp = orth(d, BASES)
Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
rho_p = -Ap / np.sqrt(Vp * E_r2)
print(f'직교화후: rho={rho_p:+.5f}  이득={K*Ap**2/Vp:+.2f}  s*={-Ap/Vp:+.4f}')
ctrl = []
for sd_ in range(20):
    rng = np.random.RandomState(18000 + sd_)
    dc = orth(rng.permutation(d0), BASES)
    Vc = float(np.mean(dc ** 2))
    if Vc > 1e-18:
        ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
ctrl = np.array(ctrl)
z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
print(f'대조군 z = {z:.1f}  ->  {"통과" if z > 2 else "허수"}')

if z <= 2:
    log('스크리닝 미통과 - 프로덕션 생략')
    sys.exit(0)

log('통과! 프로덕션(전체데이터 R/F 전문가 2모델) 학습...')
prod = {}
for name, mask_league in (('R', is_R), ('F', ~is_R)):
    trP = (cls >= 0) & mask_league
    wP = 0.5 ** ((2024.0 - season[trP]) / 2.0)
    XtrP, ctrP = X.loc[trP], cls[trP]
    n_esP = int(len(XtrP) * 0.92)
    mP = CatBoostClassifier(**CB)
    mP.fit(XtrP.iloc[:n_esP], ctrP[:n_esP], sample_weight=wP[:n_esP],
           eval_set=(XtrP.iloc[n_esP:], ctrP[n_esP:]))
    log(f'{name}전문가 프로덕션 완료 best_iter={mP.best_iteration_}')
    prod[name] = mP
joblib.dump(dict(model_R=prod['R'], model_F=prod['F'], feat_order=FEAT,
                  succ_classes=SUCC, r_value=r_value),
            'dev/mc6split_production.pkl')
log('저장 완료: dev/mc6split_production.pkl')
