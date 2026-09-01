"""XGBoost + raw ID native categorical, 우리 162피처 기반. honest fold A/C 검증.
- pitcher_id/batter_id/pitcher_team_id/batter_team_id를 XGBoost category dtype으로 직접 투입
- min_child_weight를 낮게(20 아님, 8) 줘서 표본 많은 ID가 자기 분기를 가질 여지를 줌
  (표본 적은 ID는 자연히 gain이 부족해 분기가 안 생기므로 별도 처리 불필요)
- n_estimators=3000, lr=0.01(많이 학습), early_stopping은 train 내부 마지막 8%로(Rule4 안전)
- 검증: 전체 / 고빈도-저빈도 투수 비교(가설 직접확인) / v88_final 잔차와의 클린 max-gain(중심화+무절편)
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import xgboost as xgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807
K_CONST = 1e5 / B

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
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


def build_fold(upto, vs):
    tr = season <= upto
    va = season == vs
    Xtr = X.loc[tr].copy()
    Xva = X.loc[va].copy()
    cat_p = pd.Categorical(pid[tr])
    Xtr['pitcher_id'] = cat_p
    Xva['pitcher_id'] = pd.Categorical(pid[va], categories=cat_p.categories)
    cat_b = pd.Categorical(bid[tr])
    Xtr['batter_id'] = cat_b
    Xva['batter_id'] = pd.Categorical(bid[va], categories=cat_b.categories)
    cat_pt = pd.Categorical(ptid[tr])
    Xtr['pitcher_team_id_cat'] = cat_pt
    Xva['pitcher_team_id_cat'] = pd.Categorical(ptid[va], categories=cat_pt.categories)
    cat_bt = pd.Categorical(btid[tr])
    Xtr['batter_team_id_cat'] = cat_bt
    Xva['batter_team_id_cat'] = pd.Categorical(btid[va], categories=cat_bt.categories)
    return Xtr, Xva, y[tr], y[va], tr, va


def train_eval(upto, vs, tag):
    Xtr, Xva, ytr, yva, tr, va = build_fold(upto, vs)
    w = 0.5 ** ((upto - season[tr]) / 2.0)
    n_es = int(len(Xtr) * 0.92)

    m = xgb.XGBClassifier(**PARAMS)
    m.fit(Xtr.iloc[:n_es], ytr[:n_es], sample_weight=w[:n_es],
          eval_set=[(Xtr.iloc[n_es:], ytr[n_es:])], verbose=False)
    p = np.clip(m.predict_proba(Xva)[:, 1], 0, 1)
    log(f'[{tag}] 학습완료 best_iter={m.best_iteration}')

    sc = lambda pp, msk: 1e5 * (1 - np.mean((np.clip(pp[msk], 0, 1) - yva[msk]) ** 2) / B)
    allm = np.ones(len(yva), bool)
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  단독 BSS = {sc(p, allm):.2f}')

    # 고빈도 vs 저빈도 투수 (train<=upto 기준 등장횟수)
    tr_counts = pd.Series(pid[tr]).value_counts()
    va_pid = pid[va]
    n_appear = np.array([tr_counts.get(pp, 0) for pp in va_pid])
    med = np.median(n_appear[n_appear > 0])
    hi = n_appear >= med
    lo = (n_appear > 0) & (n_appear < med)
    unseen = n_appear == 0
    for name, msk in [('고빈도(>=중앙값)', hi), ('저빈도(<중앙값)', lo), ('완전미등장', unseen)]:
        if msk.sum() < 30:
            continue
        yy = yva[msk]; pp = p[msk]
        r = yy.mean(); var_own = max(r * (1 - r), 1e-6)
        bs = np.mean((pp - yy) ** 2)
        print(f'    {name:16s} n={msk.sum():>7,}  자체BSS={1e5*(1-bs/var_own):8.1f}  편차={pp.mean()-r:+.5f}')

    return p, va


log('=== fold A (train<=2023 -> 2024) ===')
p_A, va_A = train_eval(2023, 2024, 'A')
np.save('dev/cache_xgbrawid_A.npy', p_A)

log('=== fold C (train<=2021 -> 2022) ===')
p_C, va_C = train_eval(2021, 2022, 'C')
np.save('dev/cache_xgbrawid_C.npy', p_C)

# v88_final 잔차 대비 클린 max-gain (fold A만, 캐시 있음)
log('v88_final 대비 클린검증...')
blend = np.load('dev/cache_v88_final_2024.npy')
yv = y[season == 2024]
resid = yv - blend
Xv = X.loc[season == 2024]
mth = Xv['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
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
print(f'  xgb_rawid  H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')
log('완료')
