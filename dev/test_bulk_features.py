"""'그냥 100개 파생피처를 자동생성해서 넣어보면 도움되나' 를 직접 테스트.

방법: 중요도 상위 피처들의 pairwise 비율/차/곱/로그 등을 자동생성(100+개),
      base 헤드와 동일한 단일 CatBoost로 162+100 피처 학습 vs 162만 학습을
      fold A/C에서 honest 비교. 재학습만 하면 되므로 제출 불필요.

이게 [[feature-weight-boost-rejected-twice]]와 다른 점: 그건 특정피처'가중치'를
강제로 키운 것. 이건 순수하게 '더 많은 파생피처를 후보로 추가'하는 것 -
CatBoost가 알아서 쓸모없으면 안 쓰게 둔다(강제 없음). 사용자 가설의 정확한 테스트.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier
from itertools import combinations

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
Xbase = X[FEAT].astype(np.float64)

# 중요도 상위 피처 추출 (v95의 base 관련 CatBoost가 없으면 mc5_model 등에서 근사)
imp_src = None
for k in ('mc5_model', 'midother_model', 'condball_model'):
    if v95.get(k) is not None:
        imp_src = v95[k]
        break
if imp_src is not None:
    imp = np.array(imp_src.get_feature_importance())
    order = np.argsort(-imp)
    top_feats = [FEAT[i] for i in order[:16]]
else:
    top_feats = FEAT[:16]
log(f'상위피처 16개: {top_feats}')

# pairwise 자동 파생 (비율/차/곱/로그비) -> 100개 이상
new_cols = {}
eps = 1e-6
for a, b in combinations(top_feats, 2):
    xa, xb = Xbase[a].to_numpy(), Xbase[b].to_numpy()
    new_cols[f'auto_{a}_div_{b}'] = xa / (np.abs(xb) + eps)
    new_cols[f'auto_{a}_minus_{b}'] = xa - xb
    if len(new_cols) >= 100:
        break
new_cols = dict(list(new_cols.items())[:100])
log(f'자동생성 파생피처 {len(new_cols)}개')
Xauto = pd.DataFrame(new_cols, index=Xbase.index).astype(np.float64)
Xauto = Xauto.replace([np.inf, -np.inf], 0).fillna(0)

CB = dict(iterations=1200, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='Logloss', early_stopping_rounds=50, random_seed=42)


def run(upto, vs, tag):
    tr = season <= upto
    va = season == vs
    yv = y[va]
    w = 0.5 ** ((upto - season[tr]) / 2.0)
    n_es = int(tr.sum() * 0.92)
    sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / B_)

    results = {}
    for label, Xall in [('162만(기준)', Xbase), ('162+100(자동파생)', pd.concat([Xbase, Xauto], axis=1))]:
        ts = time.time()
        m = CatBoostClassifier(**CB)
        m.fit(Xall.loc[tr].iloc[:n_es], y[tr][:n_es], sample_weight=w[:n_es],
              eval_set=(Xall.loc[tr].iloc[n_es:], y[tr][n_es:]))
        p = np.clip(m.predict_proba(Xall.loc[va])[:, 1], 0, 1)
        bss = sc(p)
        results[label] = (bss, m.best_iteration_, p)
        log(f'[{tag}/{label}] BSS={bss:.1f}  best_iter={m.best_iteration_}  '
            f'피처수={Xall.shape[1]}  ({time.time()-ts:.0f}s)')

    d = results['162+100(자동파생)'][0] - results['162만(기준)'][0]
    print(f'\n=== fold {tag} 요약 ===')
    print(f'  162만    BSS={results["162만(기준)"][0]:.1f}')
    print(f'  162+100  BSS={results["162+100(자동파생)"][0]:.1f}')
    print(f'  차이 = {d:+.1f}점 (양수면 파생피처 100개 추가가 도움)')
    return results


log('=== fold A ===')
run(2023, 2024, 'A')
log('=== fold C ===')
run(2021, 2022, 'C')
log('완료')
