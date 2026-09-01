"""v108 = v95 + XGBoost raw-ID 헤드(소량가중치, 실험적).
- build_xgb_rawid.py와 동일 레시피(n_estimators=3000/lr=0.01/depth=7/min_child_weight=8)로
  전체데이터(2019-2024)를 사용해 프로덕션용 XGB 재학습
- pitcher_id/batter_id/pitcher_team_id/batter_team_id를 XGBoost native categorical로 학습
  -> 카테고리 목록을 아티팩트에 저장(test에서 안 본 ID는 자동 NaN 처리)
- 기존 10개 헤드 가중치를 전부 (1-W_NEW)배로 비례 축소해서 재원 마련

[주의] 이 헤드는 오늘 클린검증(대조군+중심화+무절편, fold A)에서 대조군보다도 못했음
  (대조군 평균 +1.69 vs xgb_rawid 평균 +0.90, H1->H2=-1.16 부호반전).
  사용자 지시로 "검증 미통과 실험적 제출"임을 인지한 채 소량가중치(0.03)로 태움.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import xgboost as xgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

W_NEW = 0.03

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
pid = raw_all['pitcher_id'].to_numpy()
bid = raw_all['batter_id'].to_numpy()
ptid = raw_all['pitcher_team_id'].to_numpy()
btid = raw_all['batter_team_id'].to_numpy()

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT_ORDER = list(v95['feature_order'])
log(f'feature_order {len(FEAT_ORDER)}개 (v95 기준)')

PARAMS = dict(n_estimators=3000, learning_rate=0.01, max_depth=7, min_child_weight=8,
              subsample=0.9, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
              max_bin=256, tree_method='hist', random_state=42, n_jobs=-1,
              early_stopping_rounds=100, enable_categorical=True,
              objective='binary:logistic', eval_metric='logloss')

Xtr = X[FEAT_ORDER].copy()
cat_p = pd.Categorical(pid)
Xtr['pitcher_id'] = cat_p
cat_b = pd.Categorical(bid)
Xtr['batter_id'] = cat_b
cat_pt = pd.Categorical(ptid)
Xtr['pitcher_team_id_cat'] = cat_pt
cat_bt = pd.Categorical(btid)
Xtr['batter_team_id_cat'] = cat_bt

# 최근시즌(2024) 가중치 recency + 마지막 8%를 early-stopping eval로 사용(Rule4 안전, 시간순 정렬 가정)
w = 0.5 ** ((2024.0 - season) / 2.0)
n = len(y)
n_es = int(n * 0.92)

log('XGB raw-ID 전체데이터 학습...')
m = xgb.XGBClassifier(**PARAMS)
m.fit(Xtr.iloc[:n_es], y[:n_es], sample_weight=w[:n_es],
      eval_set=[(Xtr.iloc[n_es:], y[n_es:])], verbose=False)
log(f'학습완료 best_iter={m.best_iteration}')

xgbrawid_cats = dict(
    feature_order=FEAT_ORDER,
    pitcher_id_cats=list(cat_p.categories),
    batter_id_cats=list(cat_b.categories),
    pitcher_team_id_cats=list(cat_pt.categories),
    batter_team_id_cats=list(cat_bt.categories),
)

# ---------- v108 아티팩트 ----------
v108 = dict(v95)
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
print('\n=== 가중치 재배분 (기존 전부 비례축소) ===')
for k in HEADS:
    old = float(v95[f'{k}_weight'])
    new = old * (1 - W_NEW)
    v108[f'{k}_weight'] = new
    print(f'  {k:12s} {old:.4f} -> {new:.4f}')
v108['xgbrawid_weight'] = W_NEW
v108['xgbrawid_model'] = m
v108['xgbrawid_cats'] = xgbrawid_cats
tot = sum(float(v108[f'{k}_weight']) for k in HEADS) + W_NEW
print(f'  xgbrawid     0.0000 -> {W_NEW:.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v108, 'submit/model/model_artifacts_v108.pkl')
log('v108 저장 완료')
