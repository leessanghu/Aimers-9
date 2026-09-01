"""xgbunused2 = xgbunused(5개 원본플래그) + trackman 극단조잡요약 + 압축피처 꼬리플래그.

추가분:
 - tm_extreme_count: tm_*_x_lown 10개 중 |z|>2(대략 상하위 2.5%) 넘는 개수 (거칠게 카운트만)
 - inseason_is_first_appearance: 진짜 신인 첫등판 플래그(기존 미사용)
 - tail_{feat}: x_ability_here/inseason_success_smooth/inseason_cmd_index/bat_inseason_smooth
   각각의 상위3%/하위3% 이진플래그(트리가 sparse tail을 세분화 못하는 지점을 거칠게 대신 표시)

fold A 단독 학습 + 대조군(무작위 순열) z검정 + v123(현재블렌드, xgbunused+xgbrawid 포함) 기준
직교화 후 잔여신호 확인. fold C는 오늘 확인된 전역편향으로 스킵.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import xgboost as xgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

X_full = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')

TM_LOWN = ['tm_press_rel_sd_x_lown', 'tm_rel_s_mean_x_lown', 'tm_break_mag_x_lown',
           'tm_release_sd_x_lown', 'tm_speed_sd_x_lown', 'tm_rel_h_sd_x_lown',
           'tm_ivb_sd_x_lown', 'tm_rel_s_sd_x_lown', 'tm_hb_sd_x_lown', 'tm_ext_mean_x_lown']
TM_LOWN = [c for c in TM_LOWN if c in X_full.columns]

RAW_BASE = ['tm_matched', 'tm_lown_flag', 'pitcher_hand', 'form_missing', 'cat_top_bottom',
            'season', 'cat_game_type', 'inseason_is_first_appearance']
RAW_BASE = [c for c in RAW_BASE if c in X_full.columns]

TAIL_FEATS = ['x_ability_here', 'inseason_success_smooth', 'inseason_cmd_index', 'bat_inseason_smooth']
TAIL_FEATS = [c for c in TAIL_FEATS if c in X_full.columns]

Xd = X_full[RAW_BASE].astype(np.float64).copy()

# trackman 극단조잡요약: 열별 z-score 계산 후 |z|>2 개수
if TM_LOWN:
    Ztm = X_full[TM_LOWN].astype(np.float64)
    Ztm = (Ztm - Ztm.mean()) / (Ztm.std() + 1e-9)
    Xd['tm_extreme_count'] = (Ztm.abs() > 2).sum(axis=1).astype(np.float64)
    log(f'tm_extreme_count 분포: {Xd["tm_extreme_count"].value_counts().sort_index().to_dict()}')

# 압축피처 꼬리플래그 (상하위 3%)
for c in TAIL_FEATS:
    v = X_full[c].astype(np.float64)
    lo, hi = v.quantile(0.03), v.quantile(0.97)
    Xd[f'tail_{c}'] = ((v <= lo) | (v >= hi)).astype(np.float64)
    log(f'tail_{c}: 비율={Xd[f"tail_{c}"].mean()*100:.2f}%')

K_SHR = 50.0


def add_smooth(Xdf, tr_mask, key_cols):
    g_all = float(y[tr_mask].mean())
    key = pd.Series(list(zip(*[X_full[c].to_numpy()[tr_mask].astype(int) for c in key_cols])))
    ytr = y[tr_mask]
    stat = pd.DataFrame({'k': key, 'y': ytr}).groupby('k')['y'].agg(['sum', 'count'])
    smap = ((stat['sum'] + K_SHR * g_all) / (stat['count'] + K_SHR)).to_dict()
    key_all = pd.Series(list(zip(*[X_full[c].to_numpy().astype(int) for c in key_cols])))
    return key_all.map(smap).fillna(g_all).to_numpy(np.float64), smap, g_all


PARAMS = dict(n_estimators=1500, learning_rate=0.02, max_depth=5, min_child_weight=20,
              subsample=0.9, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
              max_bin=256, tree_method='hist', random_state=42, n_jobs=-1,
              early_stopping_rounds=80, objective='binary:logistic', eval_metric='logloss')

# --- fold A honest 검증 ---
upto, vs, tag = 2023, 2024, 'A'
tr = season <= upto
va = season == vs
yv = y[va]
w = 0.5 ** ((upto - season[tr]) / 2.0)

Xtr_full = Xd.copy()
sm1, _, _ = add_smooth(Xtr_full, tr, ['season', 'tm_matched'])
sm2, _, _ = add_smooth(Xtr_full, tr, ['cat_game_type', 'tm_lown_flag'])
Xtr_full['smooth_season_tmm'] = sm1
Xtr_full['smooth_gtype_lown'] = sm2

Xtr, Xva = Xtr_full.loc[tr], Xtr_full.loc[va]
n_es = int(tr.sum() * 0.92)
ts = time.time()
m = xgb.XGBClassifier(**PARAMS)
m.fit(Xtr.iloc[:n_es], y[tr][:n_es], sample_weight=w[:n_es],
      eval_set=[(Xtr.iloc[n_es:], y[tr][n_es:])], verbose=False)
p = np.clip(m.predict_proba(Xva)[:, 1], 0, 1)
np.save('dev/cache_xgbunused2_A.npy', p)
log(f'fold A 학습완료 best_iter={m.best_iteration} ({time.time()-ts:.0f}s)')
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
print(f'단독 BSS = {sc(p):.2f}')
log('완료')
