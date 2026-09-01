"""I1: adversarial validation. 2019 vs 2024 행을 구분하는 분류기 학습 ->
중요도 높은 피처 = 드리프트 중인 피처. AUC가 0.5 근처면 드리프트 미미,
1.0 근처면 두 시즌이 확연히 다른 분포(연도별 shift가 실제로 크다는 뜻).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import HistGradientBoostingClassifier

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])

FEAT_NOSEASON = [f for f in FEAT if f != 'season']  # season 자체를 빼야 '다른' 드리프트 축이 보임
# 2019는 데이터셋 첫해라 lastyear/n류 피처가 전부 0 -> 트리비얼 아티팩트(AUC=1.0 확인됨).
# 성숙한 두 해(둘 다 5년치 이력 축적)로 비교해야 진짜 행동적 드리프트를 본다.
m = (season == 2022) | (season == 2024)
Xa = X.loc[m, FEAT_NOSEASON].astype(np.float64).fillna(0.0)
ya = (season[m] == 2024).astype(int)
print(f'2022 행={(season==2022).sum():,}   2024 행={(season==2024).sum():,}')

clf = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=300,
                                     learning_rate=0.05, random_state=42)
scores = cross_val_score(clf, Xa, ya, cv=3, scoring='roc_auc', n_jobs=1)
print(f'3-fold CV AUC (2019 vs 2024 구분) = {scores.mean():.4f} ± {scores.std():.4f}')

clf.fit(Xa, ya)
imp_idx = np.argsort(-np.abs(clf.predict_proba(Xa)[:, 1] - 0.5))  # placeholder
# HGB엔 native feature_importances_가 없어서 permutation importance로 대체
from sklearn.inspection import permutation_importance
sub_idx = np.random.RandomState(0).choice(len(Xa), size=min(30000, len(Xa)), replace=False)
pi = permutation_importance(clf, Xa.iloc[sub_idx], ya[sub_idx], n_repeats=3,
                            random_state=0, n_jobs=1, scoring='roc_auc')
order = np.argsort(-pi.importances_mean)[:15]
print('\n=== 상위 15개 드리프트 피처 (season 제외, permutation importance) ===')
for i in order:
    print(f'  {FEAT_NOSEASON[i]:<38} {pi.importances_mean[i]:+.5f}')

print('\n[해석] AUC~0.5 -> 연도간 분포차 없음(드리프트 없음, season 제외하면).')
print(' AUC 높으면 상위피처가 진짜 드리프트축 -> 그 피처의 고drift 영역에서 신뢰도 낮춰볼 가치.')
print(' season 자체는 피처에서 제외하지 않았으므로 top이 season/game_month류면 당연한 것.')
