"""0-2카운트(count_state==2) 전용 투수별 EB축소 스무딩 피처를 만들어서
LightGBM(honest, train<=2023->2024)에 추가로 넣었을 때 0-2카운트 구간 성능이
plain LGBM보다 나아지는지 검증. Rule4: train 구간으로만 테이블 생성."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
import lightgbm as lgb

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807
X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
count_state = (raw_all['balls_before'] * 4 + raw_all['strikes_before']).to_numpy()
pid = raw_all['pitcher_id'].to_numpy()

tr = season <= 2023
va = season == 2024
yv = y[va]

# 0-2카운트 전용 투수별 EB축소 (train만 사용) - K를 여러 강도로 비교
is02 = (count_state == 2)
df02 = pd.DataFrame({'pid': pid[tr & is02], 'y': y[tr & is02]})
agg = df02.groupby('pid')['y'].agg(['mean', 'count'])
global02 = float(df02['y'].mean())
log(f'0-2카운트 train표본={len(df02):,}  고유투수={len(agg)}  '
    f'투수당 평균표본={len(df02)/len(agg):.1f}  global02={global02:.4f}')

FEATS = list(X.columns)
Xtr = X.copy()

SMOOTH_CANDS = {}
for K in (30.0, 200.0, 500.0, 1000.0):
    sm = (agg['count'] * agg['mean'] + K * global02) / (agg['count'] + K)
    lookup = sm.to_dict()
    arr = np.array([lookup.get(p, global02) for p in pid], dtype=np.float64)
    colname = f'count02_smooth_K{int(K)}'
    Xtr[colname] = arr
    SMOOTH_CANDS[K] = colname
    log(f'  K={K:6.0f}  피처std={arr.std():.4f}')

count02_smooth = Xtr[SMOOTH_CANDS[200.0]].to_numpy()  # 기본 비교용(K=200)
FEATS2 = FEATS + [SMOOTH_CANDS[200.0]]

w = 0.5 ** ((2023 - season) / 2.0)
ti_all = np.where(tr)[0]
n_es = int(len(ti_all) * 0.92)
ti, ei = ti_all[:n_es], ti_all[n_es:]

params = dict(objective='binary', metric='binary_logloss', learning_rate=0.03,
              num_leaves=63, max_depth=6, min_data_in_leaf=200, lambda_l2=5.0,
              verbose=-1, seed=42)


def train(feats):
    dtr = lgb.Dataset(Xtr.iloc[ti][feats], y[ti], weight=w[ti])
    dva = lgb.Dataset(Xtr.iloc[ei][feats], y[ei], weight=w[ei], reference=dtr)
    m = lgb.train(params, dtr, num_boost_round=1000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    p = np.clip(m.predict(Xtr.loc[va, feats]), 0, 1)
    return p, m


sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)
allm = np.ones(len(yv), bool)
is02_va = is02[va]
r = yv[is02_va].mean()
var_own = r * (1 - r)

log('plain LGBM(신규피처 없음) 학습...')
p_plain, m_plain = train(FEATS)
bs_plain = np.mean((p_plain[is02_va] - yv[is02_va]) ** 2)
print(f'\nplain LGBM        전체={sc(p_plain, allm):8.2f}   0-2카운트 자체BSS={1e5*(1-bs_plain/var_own):8.1f}  '
      f'편차={p_plain[is02_va].mean()-r:+.5f}')

for K in (30.0, 200.0, 500.0, 1000.0):
    feats_k = FEATS + [SMOOTH_CANDS[K]]
    log(f'LGBM + count02_smooth(K={K:.0f}) 학습...')
    p_k, m_k = train(feats_k)
    bs_k = np.mean((p_k[is02_va] - yv[is02_va]) ** 2)
    imp = m_k.feature_importance(importance_type='gain')
    fname = SMOOTH_CANDS[K]
    rank = int(np.argsort(np.argsort(-imp))[feats_k.index(fname)]) + 1
    print(f'K={K:6.0f}            전체={sc(p_k, allm):8.2f}   0-2카운트 자체BSS={1e5*(1-bs_k/var_own):8.1f}  '
          f'편차={p_k[is02_va].mean()-r:+.5f}  피처rank={rank}/{len(feats_k)}')

log('완료')
