"""v109 = v95 + XGB context-only(축소평균 제외 51피처+raw ID) 헤드, 소량가중치.
build_xgb_context_rawid.py와 동일 레시피, 전체데이터(2019-2024)로 프로덕션 재학습.
[주의] 클린검증(fold A) 대조군 +1.69 vs xgb_ctx +1.49 — 대조군 미달, 실험적 제출.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import xgboost as xgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

W_NEW = 0.03

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
season = meta['season'].to_numpy(np.float64)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
pid = raw_all['pitcher_id'].to_numpy()
bid = raw_all['batter_id'].to_numpy()
ptid = raw_all['pitcher_team_id'].to_numpy()
btid = raw_all['batter_team_id'].to_numpy()

PARAMS = dict(n_estimators=3000, learning_rate=0.01, max_depth=7, min_child_weight=8,
              subsample=0.9, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
              max_bin=256, tree_method='hist', random_state=42, n_jobs=-1,
              early_stopping_rounds=100, enable_categorical=True,
              objective='binary:logistic', eval_metric='logloss')

Xtr = X[CONTEXT_FEATS].copy()
cat_p = pd.Categorical(pid); Xtr['pitcher_id'] = cat_p
cat_b = pd.Categorical(bid); Xtr['batter_id'] = cat_b
cat_pt = pd.Categorical(ptid); Xtr['pitcher_team_id_cat'] = cat_pt
cat_bt = pd.Categorical(btid); Xtr['batter_team_id_cat'] = cat_bt

w = 0.5 ** ((2024.0 - season) / 2.0)
n = len(y)
n_es = int(n * 0.92)

log('XGB context-only 전체데이터 학습...')
m = xgb.XGBClassifier(**PARAMS)
m.fit(Xtr.iloc[:n_es], y[:n_es], sample_weight=w[:n_es],
      eval_set=[(Xtr.iloc[n_es:], y[n_es:])], verbose=False)
log(f'학습완료 best_iter={m.best_iteration}')

xgbctx_cats = dict(
    feature_order=CONTEXT_FEATS,
    pitcher_id_cats=list(cat_p.categories),
    batter_id_cats=list(cat_b.categories),
    pitcher_team_id_cats=list(cat_pt.categories),
    batter_team_id_cats=list(cat_bt.categories),
)

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v109 = dict(v95)
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
print('\n=== 가중치 재배분 (기존 전부 비례축소) ===')
for k in HEADS:
    old = float(v95[f'{k}_weight'])
    new = old * (1 - W_NEW)
    v109[f'{k}_weight'] = new
    print(f'  {k:12s} {old:.4f} -> {new:.4f}')
v109['xgbctx_weight'] = W_NEW
v109['xgbctx_model'] = m
v109['xgbctx_cats'] = xgbctx_cats
tot = sum(float(v109[f'{k}_weight']) for k in HEADS) + W_NEW
print(f'  xgbctx       0.0000 -> {W_NEW:.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v109, 'submit/model/model_artifacts_v109.pkl')
log('v109 저장 완료')
