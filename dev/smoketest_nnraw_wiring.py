"""nn_raw 배선 스모크테스트: v127에 nnraw_weight=0.01 얹어서 크래시/자료형 문제만 확인.
실제 v128 최종 가중치는 v127 실측 나온 뒤 재계산."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

v127 = joblib.load('submit/model/model_artifacts_v127.pkl')
v_test = dict(v127)
nr = joblib.load('dev/nnraw_production.pkl')
v_test['nnraw_models'] = nr['models']
v_test['nnraw_weight'] = 0.0001   # 스모크테스트 전용(합계=1 안 맞음, 실제 제출 금지) - forward 실행경로만 확인
joblib.dump(v_test, 'submit/model/model_artifacts_v999test.pkl')
print('테스트 아티팩트 저장 완료 (weight=0, 배선검증 전용)')
