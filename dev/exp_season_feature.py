"""season을 raw 숫자 피처로 쓰는 게 위험한가?
CatBoost 피처중요도 2위(12.49)인데, 트리는 학습범위 밖 숫자를 외삽 못 한다.
real test는 season=2025로 fold A/production 둘 다 한번도 못 본 값이다.
비교: season 포함 vs 제외, fold A(train<=2023->val 2024)에서.
주의: fold A 자체는 val=2024도 train 범위(2019-2023) 밖이라 이미 '외삽' 상황이다ㅡ
      그래서 이 비교가 '외삽 시 season이 해로운지'를 상당히 직접적으로 테스트한다.
"""
import numpy as np, pandas as pd, time, sys
sys.stdout.reconfigure(encoding='utf-8')
from catboost import CatBoostClassifier
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
row_num = meta['row_num'].to_numpy()
tr = season <= 2023; va = season == 2024
Xt = X.loc[tr].reset_index(drop=True); yt = y[tr]
Xv, yv = X.loc[va], y[va]
unc = 0.249807
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / unc)
rn = row_num[tr]
order = np.argsort(rn); n_es = int(len(Xt) * 0.92)
ti, ei = order[:n_es], order[n_es:]
w = 0.5 ** ((2023 - season[tr].astype(float)) / 2.0)

def fit_eval(cols, tag, Xt_src=None, Xv_src=None):
    Xt_src = Xt if Xt_src is None else Xt_src
    Xv_src = Xv if Xv_src is None else Xv_src
    Xt2 = Xt_src[cols]; Xv2 = Xv_src[cols]
    m = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
                           loss_function='Logloss', random_seed=42, verbose=False,
                           min_data_in_leaf=200, early_stopping_rounds=50)
    m.fit(Xt2.iloc[ti], yt[ti], sample_weight=w[ti], eval_set=(Xt2.iloc[ei], yt[ei]))
    p = m.predict_proba(Xv2)[:, 1]
    log(f'{tag:36s} n_feat={len(cols):3d} best_iter={m.get_best_iteration():>4}  BSS={sc(p):8.1f}')
    return p

all_cols = list(X.columns)
without_season = [c for c in all_cols if c != 'season']

# season 대신 안전한 대체: 학습범위 내로 클립한 recency 지표
Xt3 = Xt.copy(); Xv3 = Xv.copy()
cap = 2023  # train 상한
Xt3['season_capped'] = np.minimum(Xt3['season'], cap)
Xv3['season_capped'] = np.minimum(Xv3['season'], cap)  # val=2024도 cap 넘음 -> 어차피 2023으로 눌림
cols3 = without_season + ['season_capped']
BSS_WITH = 844.7  # 이전 실행에서 이미 확인됨 (season 포함, 재실행 생략)
p_capped = fit_eval(cols3, 'season_capped(상한클립)', Xt3, Xv3)
print()
print(f'클립버전 차이(클립-포함 844.7) = {sc(p_capped)-BSS_WITH:+.2f}')
