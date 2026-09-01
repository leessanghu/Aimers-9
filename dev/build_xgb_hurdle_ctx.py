"""XGB에도 '타겟분해'를 넣은 버전 (사용자 요청: 그냥 XGB가 아니라 hurdle처럼
2단계로 타겟을 쪼개고, 축소평균 없는 원시피처+raw ID로 XGB가 직접 배우게 함).

Hurdle 분해(기존 v95 hurdle헤드와 동일 정의, dev/phase86_hurdle_walkforward.py 참조):
  core_fail = 이번 투구가 reverse 또는 middle이었는가 (asof_*_rate의 누적치 diff로 복원,
              Rule4 안전: 해당 투구 발생 이후 갱신된 as-of 값에서 역산)
  1단계: m1 = P(core_fail=1)          (XGB, context+ID)
  2단계: m2 = P(y=1 | core_fail=0)    (XGB, context+ID, core_fail==0 부분집합만 학습)
  p_hur = (1-p_core) * p_snc

피처: build_xgb_context_rawid.py와 동일한 51개 원시컨텍스트 + pitcher/batter/team raw ID.
검증: honest fold A/C + v88_final 대비 클린 max-gain(중심화+무절편+대조군).
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import xgboost as xgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807

CONTEXT_FEATS = [
    'cat_top_bottom', 'cat_game_type', 'cat_base_state', 'season', 'game_month',
    'game_dayofweek', 'inning', 'balls_before', 'strikes_before', 'outs_before',
    'run_top_before', 'run_bot_before', 'run_total_before', 'score_diff_home',
    'score_diff_pitcher_team', 'runner_on_1b', 'runner_on_2b', 'runner_on_3b',
    'num_runners_on', 'home_win_expectancy', 'away_win_expectancy', 'li',
    'pitcher_hand', 'batter_hand', 'same_hand', 'count_state', 'hand_matchup',
    'flag_asof_pitcher_n_zero', 'asof_pitcher_n', 'flag_asof_batter_n_zero',
    'asof_batter_n', 'flag_asof_pitcher_pitchmix_n_zero', 'asof_pitcher_pitchmix_n',
    'flag_prev_game_missing', 'pitcher_id_count', 'batter_id_count',
    'pitcher_team_id_count', 'batter_team_id_count', 'inseason_n',
    'inseason_is_first_appearance', 'platoon_n', 'inning_n', 'pt_n',
    'x_count_pressure', 'count_n', 'vol_n_seasons', 'role_n_app', 'form_missing',
    'tm_n', 'tm_matched', 'bat_inseason_n', 'bat_ly_n', 'bplatoon_n',
]

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
pid = raw_all['pitcher_id'].to_numpy()
bid = raw_all['batter_id'].to_numpy()
ptid = raw_all['pitcher_team_id'].to_numpy()
btid = raw_all['batter_team_id'].to_numpy()

# ---- core_fail 복원 (phase86 방식, Rule4 안전: as-of 누적치 diff로 역산) ----
n_ = raw_all['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
def cnt(col):
    return np.round(raw_all[col].fillna(0).to_numpy(np.float64) * n_)
R_ = cnt('asof_pitcher_reverse_rate')
M_ = cnt('asof_pitcher_middle_rate')
ordr = raw_all.assign(row_num=np.arange(len(raw_all))).sort_values(['pitcher_id', 'row_num']).index.to_numpy()
pid_o = pid[ordr]
n_o = n_[ordr]
step = np.zeros(len(raw_all), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_o) == 1)
r_diff = np.zeros(len(raw_all)); m_diff = np.zeros(len(raw_all))
r_diff[ordr[:-1]] = np.diff(R_[ordr])
m_diff[ordr[:-1]] = np.diff(M_[ordr])
core_fail = np.where(step, ((r_diff > 0) | (m_diff > 0)).astype(np.float64), np.nan)
assert (y[step & (core_fail == 1)] == 0).all()
log(f'core_fail 복원 {step.sum():,}행  비율={np.nanmean(core_fail):.4f}')

PARAMS = dict(n_estimators=2000, learning_rate=0.02, max_depth=6, min_child_weight=15,
              subsample=0.9, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
              max_bin=256, tree_method='hist', random_state=42, n_jobs=-1,
              early_stopping_rounds=80, enable_categorical=True,
              objective='binary:logistic', eval_metric='logloss')


def add_ids(Xd, tr_m):
    cat_p = pd.Categorical(pid[tr_m])
    cat_b = pd.Categorical(bid[tr_m])
    cat_pt = pd.Categorical(ptid[tr_m])
    cat_bt = pd.Categorical(btid[tr_m])
    return dict(p=cat_p, b=cat_b, pt=cat_pt, bt=cat_bt)


def build_X(mask, cats):
    Xd = X.loc[mask, CONTEXT_FEATS].copy()
    Xd['pitcher_id'] = pd.Categorical(pid[mask], categories=cats['p'].categories)
    Xd['batter_id'] = pd.Categorical(bid[mask], categories=cats['b'].categories)
    Xd['pitcher_team_id_cat'] = pd.Categorical(ptid[mask], categories=cats['pt'].categories)
    Xd['batter_team_id_cat'] = pd.Categorical(btid[mask], categories=cats['bt'].categories)
    return Xd


def train_eval(upto, vs, tag):
    tr_m = (season <= upto) & step
    va_m = season == vs
    yv = y[va_m]
    cats = add_ids(None, tr_m)

    Xtr_core = build_X(tr_m, cats)
    ytr_core = core_fail[tr_m]
    n_es = int(len(Xtr_core) * 0.92)
    m1 = xgb.XGBClassifier(**PARAMS)
    m1.fit(Xtr_core.iloc[:n_es], ytr_core[:n_es],
           eval_set=[(Xtr_core.iloc[n_es:], ytr_core[n_es:])], verbose=False)
    Xva = build_X(va_m, cats)
    p_core = np.clip(m1.predict_proba(Xva)[:, 1], 0, 1)
    log(f'[{tag}] m1(core_fail) 학습완료 best_iter={m1.best_iteration}')

    nc_m = tr_m & (core_fail == 0)
    Xtr_nc = build_X(nc_m, cats)
    ytr_nc = y[nc_m]
    n_es2 = int(len(Xtr_nc) * 0.92)
    m2 = xgb.XGBClassifier(**PARAMS)
    m2.fit(Xtr_nc.iloc[:n_es2], ytr_nc[:n_es2],
           eval_set=[(Xtr_nc.iloc[n_es2:], ytr_nc[n_es2:])], verbose=False)
    p_snc = np.clip(m2.predict_proba(Xva)[:, 1], 0, 1)
    log(f'[{tag}] m2(success|~core_fail) 학습완료 best_iter={m2.best_iteration}')

    p_hur = np.clip((1 - p_core) * p_snc, 0, 1)
    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B)
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  hurdle_xgb 단독 BSS = {sc(p_hur):.2f}')
    return p_hur, va_m


log('=== fold A (train<=2023 -> 2024) ===')
p_A, va_A = train_eval(2023, 2024, 'A')
np.save('dev/cache_xgbhurdlectx_A.npy', p_A)

log('=== fold C (train<=2021 -> 2022) ===')
p_C, va_C = train_eval(2021, 2022, 'C')
np.save('dev/cache_xgbhurdlectx_C.npy', p_C)

log('v88_final 대비 클린검증...')
blend = np.load('dev/cache_v88_final_2024.npy')
yv = y[season == 2024]
Xv = X.loc[season == 2024]
mth = Xv['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
resid = yv - blend
sc2 = lambda pp, msk: 1e5 * (1 - np.mean((np.clip(pp[msk], 0, 1) - yv[msk]) ** 2) / B)
d = p_A - blend
rng = np.random.RandomState(8)
ctrl = rng.normal(0, d.std(), len(yv))


def honest(dd):
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        mdf = dd[fit_m].mean(); mrf = resid[fit_m].mean()
        cv = np.mean((dd[fit_m]-mdf)*(resid[fit_m]-mrf))
        vr = np.mean((dd[fit_m]-mdf)**2)
        a = cv/vr if vr > 1e-14 else 0.0
        bl = blend.copy()
        bl[ev_m] = blend[ev_m] + a*(dd[ev_m]-mdf)
        gains.append(sc2(bl, ev_m) - sc2(blend, ev_m))
    return gains


gc = honest(ctrl)
g = honest(d)
print(f'\n=== v88_final 대비 클린 max-gain (fold A) ===')
print(f'  대조군  H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
print(f'  xgb_hurdle_ctx  H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')
log('완료')
