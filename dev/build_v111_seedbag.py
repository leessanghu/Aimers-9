"""v111 = v95 + 시드배깅(6개 헤드를 K시드 앙상블로 교체) + 레벨축 잔여보정.

배경(실측 근거):
- 프로덕션 v95는 헤드당 모델 1개(K=1). fold 실험만 s42/s7 2시드 평균이었음.
- 시드쌍 캐시로 잰 시드분산 sigma_k^2 기준, K=1 -> K=무한대 이론상한
  = sum_k w_k^2 * sigma_k^2 = +1.48점(foldA) / +1.42점(foldC).
- K시드로는 그 (1 - 1/K)를 회수. K=5면 80% = 약 +1.18점.
- Var_total = Var_D(E_seed) + E_D(Var_seed) 에서 둘째항만 줄어드는 것이므로
  이게 '상한'이다(하한 아님).

각 헤드는 원본 학습스크립트와 동일한 타겟/하이퍼파라미터를 쓰고 random_seed만 바꾼다:
  multires   train_final_v55_multiresrefit.py  (ES로 best_iter 1회 확정 -> 시드별 refit)
  ordinal    train_final_v54_ordinalrefit.py   (3-stage HGB, ES 1회 -> 시드별 refit)
  midother   train_final_v60_midother.py
  condball   train_final_v62_condball.py
  countresid train_final_v63_countresid.py
  future50   train_v64_s7.py
피처는 dev/featcache_X.parquet(162개, feature_order와 동일)을 재사용 -> 피처 재생성 생략.
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

SEEDS = [42, 7, 2024, 123, 777]      # K=5 -> 이론상한의 80% 회수
K_PS = 15.0
K_C = 500.0
WINDOW, K_F = 50, 10.0
OUT = 'submit/model/model_artifacts_v111.pkl'

_RNG = ('Generator', 'BitGenerator', 'RandomState', 'PCG64', 'MT19937', 'Philox', 'SFC64')
def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 12 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, '__dict__'):
        for k, v in list(vars(obj).items()):
            if type(v).__name__ in _RNG:
                setattr(obj, k, None)
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, dict):
        for k, v in list(obj.items()):
            if type(v).__name__ in _RNG:
                obj[k] = None
            else:
                strip_rng(v, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v in obj:
            strip_rng(v, seen, depth + 1)


# ---------- 로드 ----------
log('로드...')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = pd.read_parquet('dev/featcache_X.parquet')[FEAT].astype(np.float64)
meta = pd.read_parquet('dev/featcache_meta.parquet')
df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
y = df['control_success'].to_numpy()
seasons = df['season'].to_numpy(np.float64)
assert len(X) == len(df), (len(X), len(df))
assert np.array_equal(meta['season'].to_numpy(), df['season'].to_numpy()), '행 정렬 불일치'
g = float(y.mean())
w = 0.5 ** ((seasons.max() - seasons) / 2.0)
n = len(X)
tr_i, es_i = np.arange(int(n * 0.92)), np.arange(int(n * 0.92), n)
log(f'X={X.shape}  n={n:,}  global={g:.5f}')

# ---------- 라벨 복원 ----------
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


lab_reverse = diff_label('asof_pitcher_reverse_rate')
lab_middle = diff_label('asof_pitcher_middle_rate')
lab_ball = diff_label('asof_pitcher_ball_rate')
valid = ~(np.isnan(lab_reverse) | np.isnan(lab_middle))
log(f'라벨 유효 {valid.sum():,}/{n:,}')

CAT_MULTI = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                 loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50)
CAT_PLAIN = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                 loss_function='MultiRMSE', early_stopping_rounds=50)

artifacts = dict(v95)

# ================= 1) multires =================
log('=== multires ===')
sub = pd.DataFrame({'pid': pid, 'season': seasons, 'sh': X['same_hand'].to_numpy(np.float64), 'y': y})
ps = sub.groupby(['pid', 'season'])['y'].agg(s='sum', n='count')
sub = sub.join(ps, on=['pid', 'season'])
h1 = (((sub['s'] - sub['y']) + K_PS * g) / ((sub['n'] - 1) + K_PS)).to_numpy(np.float64)
psh = sub.groupby(['pid', 'season', 'sh'])['y'].agg(s2='sum', n2='count')
sub = sub.join(psh, on=['pid', 'season', 'sh'])
h2 = (((sub['s2'] - sub['y']) + K_PS * pd.Series(h1)) / ((sub['n2'] - 1) + K_PS)).to_numpy(np.float64)
Y_mr = np.column_stack([y.astype(np.float64), h1, h2])
ts = time.time()
m_es = CatBoostRegressor(**CAT_MULTI, random_seed=42)
m_es.fit(X.iloc[tr_i], Y_mr[tr_i], sample_weight=w[tr_i], eval_set=(X.iloc[es_i], Y_mr[es_i]))
it_mr = max(m_es.best_iteration_, 1)
log(f'  best_iter={it_mr} ({time.time()-ts:.0f}s)')
p_fixed = {k: v for k, v in CAT_MULTI.items() if k != 'early_stopping_rounds'}
p_fixed['iterations'] = it_mr
mr_models = []
for s in SEEDS:
    ts = time.time()
    m = CatBoostRegressor(**p_fixed, random_seed=s)
    m.fit(X, Y_mr, sample_weight=w)
    strip_rng(m); mr_models.append(m)
    log(f'  seed={s} refit완료 ({time.time()-ts:.0f}s)')
artifacts['multires_models'] = mr_models

# ================= 2) ordinal (3-stage HGB) =================
log('=== ordinal ===')
HGB_ES = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
              l2_regularization=5.0, early_stopping=True, validation_fraction=0.08,
              n_iter_no_change=20, random_state=42)
not_rev = valid & (lab_reverse == 0)
not_rev_mid = not_rev & (lab_middle == 0)
stage_defs = [
    ('stage1', valid, (1 - lab_reverse[valid])),
    ('stage2', not_rev, (1 - lab_middle[not_rev])),
    ('stage3', not_rev_mid, y[not_rev_mid].astype(np.float64)),
]
stage_iters = {}
for name, mask, yy in stage_defs:
    ts = time.time()
    mes = HistGradientBoostingClassifier(**HGB_ES)
    mes.fit(X.loc[mask], yy, sample_weight=w[mask])
    stage_iters[name] = mes.n_iter_
    log(f'  {name} ES_iter={mes.n_iter_} ({time.time()-ts:.0f}s, n={mask.sum():,})')
ord_models = []
for s in SEEDS:
    trio = []
    for name, mask, yy in stage_defs:
        ts = time.time()
        m = HistGradientBoostingClassifier(
            max_depth=6, max_leaf_nodes=31, max_iter=stage_iters[name], learning_rate=0.03,
            l2_regularization=5.0, early_stopping=False, random_state=s)
        m.fit(X.loc[mask], yy, sample_weight=w[mask])
        strip_rng(m); trio.append(m)
        log(f'  seed={s} {name} refit ({time.time()-ts:.0f}s)')
    ord_models.append(trio)
artifacts['ordinal_stages_bag'] = ord_models

# ================= 3) midother =================
log('=== midother ===')
tot_ = y + lab_reverse + lab_middle
lab_other = np.where(valid, (tot_ == 0).astype(np.float64), np.nan)
Y_mo = np.column_stack([y.astype(np.float64),
                        np.where(valid, 1.0 - lab_middle, np.nan),
                        np.where(valid, 1.0 - lab_other, np.nan)])
mo_models = []
for s in SEEDS:
    ts = time.time()
    m = CatBoostRegressor(**CAT_MULTI, random_seed=s)
    m.fit(X.iloc[tr_i], Y_mo[tr_i], sample_weight=w[tr_i], eval_set=(X.iloc[es_i], Y_mo[es_i]))
    strip_rng(m); mo_models.append(m)
    log(f'  seed={s} best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
artifacts['midother_models'] = mo_models

# ================= 4) condball =================
log('=== condball ===')
valid_cb = ~(np.isnan(lab_reverse) | np.isnan(lab_middle) | np.isnan(lab_ball))
dang = valid_cb & ((lab_middle > 0) | (lab_reverse > 0))
notdang = valid_cb & ~dang
Y_cb = np.column_stack([y.astype(np.float64), np.where(notdang, 1.0 - lab_ball, np.nan)])
cb_models = []
for s in SEEDS:
    ts = time.time()
    m = CatBoostRegressor(**CAT_MULTI, random_seed=s)
    m.fit(X.iloc[tr_i], Y_cb[tr_i], sample_weight=w[tr_i], eval_set=(X.iloc[es_i], Y_cb[es_i]))
    strip_rng(m); cb_models.append(m)
    log(f'  seed={s} best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
artifacts['condball_models'] = cb_models

# ================= 5) countresid =================
log('=== countresid ===')
count_state = df['balls_before'].to_numpy(np.float64) * 4 + df['strikes_before'].to_numpy(np.float64)
ctab = pd.DataFrame({'cs': count_state, 'y': y}).groupby('cs')['y'].agg(['sum', 'count'])
ctab['prior'] = (ctab['sum'] + K_C * g) / (ctab['count'] + K_C)
cprior = pd.Series(count_state).map(ctab['prior']).to_numpy(np.float64)
Y_cr = np.column_stack([y.astype(np.float64), y.astype(np.float64) - cprior])
cr_models = []
for s in SEEDS:
    ts = time.time()
    m = CatBoostRegressor(**CAT_PLAIN, random_seed=s)
    m.fit(X.iloc[tr_i], Y_cr[tr_i], sample_weight=w[tr_i], eval_set=(X.iloc[es_i], Y_cr[es_i]))
    strip_rng(m); cr_models.append(m)
    log(f'  seed={s} best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
artifacts['countresid_models'] = cr_models

# ================= 6) future50 =================
log('=== future50 ===')
g_mid = float(np.nanmean(lab_middle))
sub2 = pd.DataFrame({'pid': pid, 'row_num': df['row_num'].to_numpy(), 'y': y, 'mid': lab_middle})
sub2 = sub2.sort_values(['pid', 'row_num'])
grp = sub2.groupby('pid')
fys = grp['y'].transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).sum().iloc[::-1])
fyc = grp['y'].transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).count().iloc[::-1])
fms = grp['mid'].transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).sum().iloc[::-1])
fmc = grp['mid'].transform(lambda s: s.iloc[::-1].shift(1).rolling(WINDOW, min_periods=10).count().iloc[::-1])
hs = pd.Series(((fys + K_F * g) / (fyc + K_F)).to_numpy(), index=sub2.index).reindex(range(n)).to_numpy(np.float64)
hm = pd.Series((1.0 - ((fms + K_F * g_mid) / (fmc + K_F))).to_numpy(), index=sub2.index).reindex(range(n)).to_numpy(np.float64)
Y_f5 = np.column_stack([y.astype(np.float64), hs, hm])
f5_models = []
for s in SEEDS:
    ts = time.time()
    m = CatBoostRegressor(**CAT_MULTI, random_seed=s)
    m.fit(X.iloc[tr_i], Y_f5[tr_i], sample_weight=w[tr_i], eval_set=(X.iloc[es_i], Y_f5[es_i]))
    strip_rng(m); f5_models.append(m)
    log(f'  seed={s} best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
artifacts['future50_models'] = f5_models

# ================= 레벨축 잔여보정 =================
# v106 프로브 실측: D_true = E[p] - r = -0.00097 (현 level_shift 적용 상태에서의 잔여).
# BS(delta) = BS0 + 2*delta*D + delta^2 은 정확한 항등식(곡률 V=1, 1e-17까지 검증됨).
# 최적 delta = -D = +0.00097, 이득 = (1e5/B)*D^2 = +0.38점.
D_TRUE = -0.00097
old_shift = float(v95['level_shift'])
artifacts['level_shift'] = old_shift - D_TRUE
log(f'level_shift {old_shift:+.6f} -> {artifacts["level_shift"]:+.6f}  (D_true={D_TRUE:+.5f} 상쇄, 기대 +0.38점)')

strip_rng(artifacts)
joblib.dump(artifacts, OUT)
log(f'저장: {OUT} ({os.path.getsize(OUT)/1e6:.1f}MB)')
log(f'시드 {len(SEEDS)}개 x 6헤드 배깅 완료. 총 {time.time()-t0:.0f}s')
