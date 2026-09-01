"""v105 = 프로덕션과 동일 레시피(단, 시간상 일부 헤드는 단순화)로 5-fold 교차학습해
전체 train(2019-2024)에 대한 '진짜 자기 자신 OOF'를 얻고, 그걸로 헤드별 isotonic 맵을
학습. cross-model-calibration-mismatch(v104 실패원인) 재발 방지 - 여기서 만드는 OOF는
production과 동일한 하이퍼파라미터/피처/타깃 레시피를 쓰므로 분포가 일치함.

우선순위(블렌드 비중 순): hurdle > mc5 > base > midother > ordinal > ingame >
condball > countresid > future50 > multires
6시간 내 다 못 끝내도 헤드/폴드 단위로 체크포인트 저장하므로 중간에 죽어도 재사용 가능.

단순화: multires/countresid/future50/ingame/base는 원래 multi-task 보조타깃이
있지만(다른 헤드와 분간을 위해 필요) 시간상 y 단일회귀로 근사. midother/condball은
cls5에서 싸게 복원 가능해 원래 보조타깃 그대로 사용. hurdle/ordinal/mc5는 원래
레시피 그대로(이미 오늘 여러 번 확인된 구조).
"""
import os, sys, time, json
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
def log(m): print(f'[{time.time()-t0:8.0f}s] {m}', flush=True)

CKPT_DIR = 'dev/v105_ckpt'
os.makedirs(CKPT_DIR, exist_ok=True)
PROGRESS_FILE = f'{CKPT_DIR}/progress.json'

log('데이터 로드...')
X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
cls5 = np.load('dev/cls5_labels.npy')
pt = np.load('dev/pitchtype_labels.npy')
n = len(y)
w = 0.5 ** ((2024.0 - season) / 2.0)
log(f'전체 n={n:,}')

valid5 = cls5 >= 0
is_mid = (cls5 == 0)
is_rev = (cls5 == 1)
is_ball = (cls5 == 2)
is_other = (cls5 == 4)
notdang = cls5 >= 2  # not middle/reverse

cls11 = np.full(n, -1, dtype=np.int64)
v11 = valid5 & (pt >= 0)
nd11 = v11 & (cls5 >= 2)
cls11[nd11] = (cls5[nd11] - 2) * 3 + pt[nd11]
cls11[v11 & is_mid] = 9
cls11[v11 & is_rev] = 10

K = 3  # hurdle fold0 실측(22분/1개헤드)으로 재추정한 결과 5-fold는 6시간에 못 맞음.
# K를 줄여도 전체 fold를 다 돌면 OOF coverage는 항상 100% - 다만 폴드당 학습표본이
# 80%->66%로 줄어 모델이 살짝 노이즈해짐(캘리브레이션 맵 용도로는 허용 가능한 트레이드오프).
kf = KFold(n_splits=K, shuffle=True, random_state=42)
fold_id = np.zeros(n, dtype=int)
for i, (_, te_idx) in enumerate(kf.split(np.arange(n))):
    fold_id[te_idx] = i
log(f'{K}-fold 분할 완료')

CAT_REG = dict(iterations=700, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
               loss_function='RMSE', early_stopping_rounds=40)
CAT_REG_MULTI = dict(iterations=700, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                      loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=40)
CAT_CLS = dict(iterations=700, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
               loss_function='Logloss', early_stopping_rounds=40)
HGB_CLS = dict(max_depth=6, max_leaf_nodes=31, max_iter=400, learning_rate=0.03,
               l2_regularization=5.0, early_stopping=True, validation_fraction=0.08)
MC5_CFG = dict(iterations=700, learning_rate=0.05, depth=6, l2_leaf_reg=5.0, verbose=0,
               loss_function='MultiClass', classes_count=11, early_stopping_rounds=40)

HEAD_ORDER = ['mc5', 'midother', 'condball', 'ordinal',
              'hurdle', 'base', 'ingame', 'countresid', 'future50', 'multires']

def load_progress():
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}

def save_progress(p):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(p, f)

def oof_path(head):
    return f'{CKPT_DIR}/oof_{head}.npy'

def train_fold_head(head, fold):
    tr = fold_id != fold
    te = fold_id == fold
    pred_te = np.full(te.sum(), np.nan)
    te_idx = np.where(te)[0]

    if head == 'base':
        m = CatBoostRegressor(**CAT_REG, random_seed=42)
        m.fit(X.loc[tr], y[tr], sample_weight=w[tr])
        pred_te = np.clip(m.predict(X.loc[te]), 0, 1)

    elif head == 'hurdle':
        v1 = tr & valid5
        core_fail = ((is_mid | is_rev).astype(np.float64))
        m1 = CatBoostClassifier(**CAT_CLS, random_seed=42)
        m1.fit(X.loc[v1], core_fail[v1], sample_weight=w[v1])
        not_cf = v1 & (core_fail == 0)
        m2 = CatBoostClassifier(**CAT_CLS, random_seed=42)
        m2.fit(X.loc[not_cf], y[not_cf], sample_weight=w[not_cf])
        p_cf = m1.predict_proba(X.loc[te])[:, 1]
        p_snc = m2.predict_proba(X.loc[te])[:, 1]
        pred_te = np.clip((1 - p_cf) * p_snc, 0, 1)

    elif head == 'multires':
        m = CatBoostRegressor(**CAT_REG, random_seed=43)
        m.fit(X.loc[tr], y[tr], sample_weight=w[tr])
        pred_te = np.clip(m.predict(X.loc[te]), 0, 1)

    elif head == 'ordinal':
        v1 = tr & valid5
        m1 = HistGradientBoostingClassifier(**HGB_CLS, random_state=42)
        m1.fit(X.loc[v1], (1 - is_rev[v1].astype(np.float64)), sample_weight=w[v1])
        not_rev = v1 & (~is_rev)
        m2 = HistGradientBoostingClassifier(**HGB_CLS, random_state=42)
        m2.fit(X.loc[not_rev], (1 - is_mid[not_rev].astype(np.float64)), sample_weight=w[not_rev])
        not_rev_mid = not_rev & (~is_mid)
        m3 = HistGradientBoostingClassifier(**HGB_CLS, random_state=42)
        m3.fit(X.loc[not_rev_mid], y[not_rev_mid], sample_weight=w[not_rev_mid])
        po1 = m1.predict_proba(X.loc[te])[:, 1]
        po2 = m2.predict_proba(X.loc[te])[:, 1]
        po3 = m3.predict_proba(X.loc[te])[:, 1]
        pred_te = np.clip(po1 * po2 * po3, 0, 1)

    elif head == 'midother':
        h1 = np.where(valid5, (1.0 - is_mid.astype(np.float64)), np.nan)
        h2 = np.where(valid5, (1.0 - is_other.astype(np.float64)), np.nan)
        Ymat = np.column_stack([y, h1, h2])
        m = CatBoostRegressor(**CAT_REG_MULTI, random_seed=42)
        m.fit(X.loc[tr], Ymat[tr], sample_weight=w[tr])
        pred_te = np.clip(m.predict(X.loc[te])[:, 0], 0, 1)

    elif head == 'condball':
        h_cb = np.where(notdang, (1.0 - is_ball.astype(np.float64)), np.nan)
        Ymat = np.column_stack([y, h_cb])
        m = CatBoostRegressor(**CAT_REG_MULTI, random_seed=42)
        m.fit(X.loc[tr], Ymat[tr], sample_weight=w[tr])
        pred_te = np.clip(m.predict(X.loc[te])[:, 0], 0, 1)

    elif head == 'countresid':
        m = CatBoostRegressor(**CAT_REG, random_seed=44)
        m.fit(X.loc[tr], y[tr], sample_weight=w[tr])
        pred_te = np.clip(m.predict(X.loc[te]), 0, 1)

    elif head == 'future50':
        m = CatBoostRegressor(**CAT_REG, random_seed=45)
        m.fit(X.loc[tr], y[tr], sample_weight=w[tr])
        pred_te = np.clip(m.predict(X.loc[te]), 0, 1)

    elif head == 'mc5':
        v1 = tr & (cls11 >= 0)
        m = CatBoostClassifier(**MC5_CFG, random_seed=42)
        m.fit(X.loc[v1], cls11[v1], sample_weight=w[v1])
        proba11 = m.predict_proba(X.loc[te])
        succ = np.zeros(11)
        for c in range(11):
            mtr = v1 & (cls11 == c)
            succ[c] = y[mtr].mean() if mtr.sum() > 0 else 0.5
        pred_te = np.clip(proba11 @ succ, 0, 1)

    elif head == 'ingame':
        m = CatBoostRegressor(**CAT_REG, random_seed=46)
        m.fit(X.loc[tr], y[tr], sample_weight=w[tr])
        pred_te = np.clip(m.predict(X.loc[te]), 0, 1)

    return te_idx, pred_te

progress = load_progress()
for head in HEAD_ORDER:
    oof = np.full(n, np.nan)
    p = oof_path(head)
    if os.path.exists(p):
        oof = np.load(p)
    done_folds = set(progress.get(head, []))
    for fold in range(K):
        if fold in done_folds:
            continue
        ts = time.time()
        te_idx, pred_te = train_fold_head(head, fold)
        oof[te_idx] = pred_te
        np.save(p, oof)
        done_folds.add(fold)
        progress[head] = sorted(done_folds)
        save_progress(progress)
        log(f'{head} fold{fold} 완료 ({time.time()-ts:.0f}s)')
    log(f'=== {head} 전체 5-fold 완료 ===')

log('모든 헤드 완료. isotonic 맵 학습...')
iso_maps = {}
for head in HEAD_ORDER:
    p = oof_path(head)
    if not os.path.exists(p):
        continue
    oof = np.load(p)
    m = ~np.isnan(oof)
    if m.sum() < 1000:
        continue
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(oof[m], y[m])
    iso_maps[head] = iso
    log(f'{head} isotonic map 학습 완료 (n={m.sum():,})')

joblib.dump(iso_maps, f'{CKPT_DIR}/iso_maps_honest.pkl')
log(f'iso_maps 저장 완료 ({len(iso_maps)}개 헤드)')
log('전체 완료')
