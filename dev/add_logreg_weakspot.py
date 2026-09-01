"""로지스틱회귀(완전히 다른 함수형태) 동일조건 비교 추가. 트리 계열 아님.
결측 있는 피처가 있어 median 대치 + 표준화 후 학습."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807
X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
count_state = (raw_all['balls_before'] * 4 + raw_all['strikes_before']).to_numpy()
n_pitcher = np.nan_to_num(raw_all['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)

tr = season <= 2023
va = season == 2024
yv = y[va]
w = 0.5 ** ((2023 - season) / 2.0)

log('결측대치+표준화...')
imp = SimpleImputer(strategy='median')
Xtr_imp = imp.fit_transform(X.loc[tr])
Xva_imp = imp.transform(X.loc[va])
sc_ = StandardScaler()
Xtr_s = sc_.fit_transform(Xtr_imp)
Xva_s = sc_.transform(Xva_imp)

log('로지스틱회귀 학습...')
m = LogisticRegression(max_iter=2000, C=1.0)
m.fit(Xtr_s, y[tr], sample_weight=w[tr])
p_logreg = np.clip(m.predict_proba(Xva_s)[:, 1], 0, 1)
log('완료')

is_rookie = n_pitcher[va] < 0.5
is02 = count_state[va] == 2
allm = np.ones(len(yv), bool)
groups = [('전체', allm), ('완전신인(n=0)', is_rookie), ('0-2카운트', is02)]

print(f'\n=== LogisticRegression ===')
for gname, m2 in groups:
    yy = yv[m2]; pp = p_logreg[m2]
    r = yy.mean(); var_own = max(r * (1 - r), 1e-6)
    bs = np.mean((pp - yy) ** 2)
    bias = pp.mean() - r
    print(f'  {gname:16s} 자체BSS={1e5*(1-bs/var_own):8.1f}  편차={bias:+.5f}  n={m2.sum():,}')

np.save('dev/cache_logreg_2024.npy', p_logreg)
log('저장 완료')
