"""HGB 계열(base_hgb, hurdle의 core_fail/succ_nc, ordinal 3단)의 permutation importance.
production 모델 그대로, 표본 5만행으로 근사(순위 파악 목적, 정밀 이득추정 아님)."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
feats = v88['feature_order']
X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)

rng = np.random.RandomState(0)
idx = rng.choice(len(y), size=50000, replace=False)
Xs = X.iloc[idx][feats]
ys = y[idx]

DEV_FEATS = ['form5_success', 'form3_success', 'x_prev5_minus_career', 'ly_minus_career',
             'x_kal_minus_career', 'x_prev1_minus_prev5', 'form1_success', 'bat_middle_minus_career',
             'form_accel', 'score_diff_pitcher_team', 'form5_middle', 'form_3_minus_5', 'form3_middle',
             'inseason_middle_minus_career', 'form1_middle', 'form_reliability', 'form_1_minus_3',
             'bat_inseason_minus_career', 'diff_success_rate', 'score_diff_home', 'diff_middle_rate',
             'form_missing']

class ProbaWrap:
    def __init__(self, model, idx1=1):
        self.model = model
        self.idx1 = idx1
    def predict_proba(self, X):
        p = self.model.predict_proba(X)
        return p

def scorer(estimator, X, y_true):
    p = estimator.predict_proba(X)[:, 1]
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -log_loss(y_true, p)

heads = {
    'base_hgb0': v88['hgbs'][0],
    'hurdle_corefail0': v88['core_fail_models'][0],
    'hurdle_succnc0': v88['succ_nc_models'][0],
    'ordinal_stage1': v88['ordinal_stage1'],
    'ordinal_stage2': v88['ordinal_stage2'],
    'ordinal_stage3': v88['ordinal_stage3'],
}

results = {}
for name, model in heads.items():
    log(f'{name} permutation importance 계산...')
    r = permutation_importance(model, Xs, ys, scoring=scorer, n_repeats=3, random_state=0, n_jobs=1)
    imp = r.importances_mean
    order = np.argsort(imp)[::-1]
    rank_of = {feats[i]: list(order).index(i) + 1 for i in range(len(feats))}
    results[name] = (imp, rank_of)
    top5 = [feats[i] for i in order[:5]]
    log(f'  {name} top5: {top5}')

print()
print('=== 편차/트렌드 계열 피처의 HGB 헤드별 순위 (총 162개 중) ===')
print(f'{"피처":32s} ' + ' '.join(f'{n:>16s}' for n in heads))
for f in DEV_FEATS:
    if f not in feats:
        continue
    row = []
    for name in heads:
        row.append(f'{results[name][1].get(f, -1):>16d}')
    print(f'{f:32s} ' + ' '.join(row))

print()
print('=== 참고: 각 HGB헤드 전체 top10 ===')
for name in heads:
    imp, rank_of = results[name]
    order = np.argsort(imp)[::-1]
    print(f'\n{name}:')
    for i in order[:10]:
        print(f'  {feats[i]:32s} {imp[i]:.5f}')
log('완료')
