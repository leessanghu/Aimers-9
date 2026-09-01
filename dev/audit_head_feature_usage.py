"""이미 학습된 CatBoost 헤드들(v95/v112)의 feature_importance를 재학습 없이 조회.
어떤 피처를 헤드들이 많이/적게 읽는지 지도화 -> 못 읽는 피처(전 헤드 공통 저importance) 탐색."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])

CAT_HEADS = ['mc5_model', 'condball_model', 'countresid_model', 'future50_model',
             'midother_model', 'multires_model', 'ingame_model']

imp = {}
for hk in CAT_HEADS:
    m = v95[hk]
    fi = m.get_feature_importance()
    nm = hk.replace('_model', '')
    imp[nm] = pd.Series(fi, index=FEAT)
    print(f'{nm:<14} 로드완료 (n_features={len(fi)})')

# mc6 (v112에 저장돼 있으면 로드)
try:
    v112 = joblib.load('submit/model/model_artifacts_v112.pkl')
    m6 = v112['mc6pure_model']
    imp['mc6'] = pd.Series(m6.get_feature_importance(), index=FEAT)
    print('mc6            로드완료')
except Exception as e:
    print(f'mc6 로드 실패: {e}')

IMP = pd.DataFrame(imp)
IMP['평균'] = IMP.mean(axis=1)
IMP['최대'] = IMP.max(axis=1)
IMP_sorted = IMP.sort_values('평균', ascending=False)

print(f'\n{"="*100}\n=== 전 헤드 평균 importance 상위 20 ===\n{"="*100}')
print(IMP_sorted.head(20).round(2).to_string())

print(f'\n{"="*100}\n=== 전 헤드 평균 importance 하위 20 (다들 안 읽는 피처) ===\n{"="*100}')
print(IMP_sorted.tail(20).round(3).to_string())

print(f'\n{"="*100}\n=== "한 헤드만 크게 쓰는" 피처 top15 (최대-평균 격차 큰 순, 헤드별 특화신호) ===\n{"="*100}')
IMP['격차'] = IMP['최대'] - IMP['평균']
spec = IMP.sort_values('격차', ascending=False).head(15)
for feat in spec.index:
    row = IMP.loc[feat, [c for c in imp.keys()]]
    top_head = row.idxmax()
    print(f'  {feat:<40} 평균={IMP.loc[feat,"평균"]:6.2f}  최대={IMP.loc[feat,"최대"]:6.2f}({top_head})')

IMP.to_csv('dev/head_feature_importance.csv')
print(f'\n전체 표는 dev/head_feature_importance.csv 에 저장')
