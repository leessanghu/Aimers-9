"""half-life 대실험: 전 헤드가 서있는 recency 가중치(hl=2.0, v20시절 로컬로 결정)를 재검증.

근거: season 피처 제거 = 로컬 -435 (시간축이 이 문제 최대 신호). 성공률 연 -1.5%p
  단조하락, 테스트는 2025 외삽. hl은 드리프트 적응을 통제하는 유일한 전역 레버인데
  실측된 적 없음.

1단계: base-HGB(빠름)로 hl in {1.0, 2.0, 4.0} fold A 방향탐색.
2단계: 이긴 hl로 mc6(지배헤드, 6클래스 CatBoost) fold A 재학습.
       기존 mc6에 직교화 + 순열대조군 z. (d = mc6_hl - blend 는 사실상
       '드리프트 적응 방향' - 기존 mc6와의 차이가 주성분)
3단계: z>2 통과시 프로덕션(전체데이터) 자동 학습.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
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

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'pitcher_id', 'asof_pitcher_n',
                          'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate'])
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

tr = season <= 2023
va = season == 2024
yv = y_all[va]
Xtr_all, Xva = X.loc[tr], X.loc[va]
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)

# ---------- 1단계: base-HGB hl 스윕 ----------
log('=== 1단계: base-HGB half-life 방향탐색 ===')
HGB = dict(max_depth=6, max_leaf_nodes=31, max_iter=400, learning_rate=0.05,
           l2_regularization=5.0, early_stopping=True, validation_fraction=0.08,
           n_iter_no_change=20, random_state=42)
results_hl = {}
for hl in (1.0, 2.0, 4.0):
    w = 0.5 ** ((2023.0 - season[tr]) / hl)
    m = HistGradientBoostingClassifier(**HGB)
    m.fit(Xtr_all, y_all[tr], sample_weight=w)
    p = np.clip(m.predict_proba(Xva)[:, 1], 0, 1)
    results_hl[hl] = sc(p)
    log(f'  hl={hl}: BSS={results_hl[hl]:.2f}')
best_hl = max(results_hl, key=results_hl.get)
print(f'\n1단계 결과: ' + '  '.join(f'hl={h}:{v:.1f}' for h, v in results_hl.items()))
print(f'승자 = hl={best_hl}  (기존 2.0 대비 {results_hl[best_hl]-results_hl[2.0]:+.2f})')

if best_hl == 2.0:
    log('기존 hl=2.0이 최선 - 드리프트 레버 없음, 종료')
    sys.exit(0)

# ---------- 2단계: mc6를 best_hl로 재학습 (fold A) ----------
log(f'=== 2단계: mc6 hl={best_hl} 재학습 (fold A) ===')
CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=6, early_stopping_rounds=50,
          random_seed=42)
tr6 = tr & (cls >= 0)
w6 = 0.5 ** ((2023.0 - season[tr6]) / best_hl)
Xtr6, ctr6 = X.loc[tr6], cls[tr6]
n_es = int(len(Xtr6) * 0.92)
m6 = CatBoostClassifier(**CB)
m6.fit(Xtr6.iloc[:n_es], ctr6[:n_es], sample_weight=w6[:n_es],
       eval_set=(Xtr6.iloc[n_es:], ctr6[n_es:]))
proba = m6.predict_proba(Xva)
p_mc6hl = np.clip(proba[:, SUCC].sum(axis=1), 0, 1)
np.save(f'dev/cache_mc6hl{best_hl}_A.npy', p_mc6hl)
log(f'학습완료 best_iter={m6.best_iteration_}')

# ---------- 스크리닝 ----------
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


print(f'\nmc6_hl{best_hl} 단독 BSS = {sc(p_mc6hl):.2f} (기존 mc6 = {sc(COMPS["mc6"]):.2f})')
print(f'기존 mc6와 예측상관 = {np.corrcoef(p_mc6hl, COMPS["mc6"])[0,1]:.4f}')
d = p_mc6hl - blend; d0 = d - d.mean()
V = float(np.mean(d0 ** 2)); A = float(np.mean(d0 * (blend - yv)))
print(f'원본:    rho={-A/np.sqrt(V*E_r2):+.5f}  s*={-A/V:+.4f}')
dp = orth(d, BASES)
Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
rho_p = -Ap / np.sqrt(Vp * E_r2)
print(f'직교화후: rho={rho_p:+.5f}  이득={K*Ap**2/Vp:+.2f}  s*={-Ap/Vp:+.4f}')
ctrl = []
for sd_ in range(20):
    rng = np.random.RandomState(17000 + sd_)
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

# ---------- 3단계: 프로덕션 ----------
log(f'=== 3단계: mc6 hl={best_hl} 프로덕션 (전체데이터) ===')
trF = cls >= 0
wF = 0.5 ** ((2024.0 - season[trF]) / best_hl)
XtrF, ctrF = X.loc[trF], cls[trF]
n_esF = int(len(XtrF) * 0.92)
mF = CatBoostClassifier(**CB)
mF.fit(XtrF.iloc[:n_esF], ctrF[:n_esF], sample_weight=wF[:n_esF],
       eval_set=(XtrF.iloc[n_esF:], ctrF[n_esF:]))
log(f'프로덕션 학습완료 best_iter={mF.best_iteration_}')
joblib.dump(dict(model=mF, feat_order=FEAT, succ_classes=SUCC, half_life=best_hl),
            'dev/mc6hl_production.pkl')
log('저장 완료: dev/mc6hl_production.pkl')
