"""v131에서 weight=0인 죽은 모델 객체 제거 - 파일 축소(업로드 리스크 최소화)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib, os

v131 = joblib.load('submit/model/model_artifacts_v131.pkl')
DEAD_KEYS = ['mc6aux_model', 'mc6aux_feat_order',
             'n1_models', 'n1_raw18_feats',
             'nnraw_models',
             'zoneintent_model', 'zoneintent_succ_classes',
             'et_model',
             'lgbmmc6_model',  # 혹시 있으면
             ]
removed = []
for k in DEAD_KEYS:
    if k in v131:
        del v131[k]
        removed.append(k)
print('제거:', removed)
# 가중치 키는 0으로 유지(스크립트가 .get으로 안전 처리)
joblib.dump(v131, 'submit/model/model_artifacts_v131.pkl')
print(f'크기: {os.path.getsize("submit/model/model_artifacts_v131.pkl")/1e6:.1f}MB')
