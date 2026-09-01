"""v132 = v131(xu고정 최적점, 안전판) + F전문가 라우팅(F행의 mc6 예측만 교체).
가중치 변경 없음 - mc6 슬롯 내부에서 F행 예측만 개선. fold A F행 직접비교로 검증됨."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

v131 = joblib.load('submit/model/model_artifacts_v131.pkl')
v132 = dict(v131)

fx = joblib.load('dev/fexpert_production.pkl')
v132['fexpert_model'] = fx['model']
v132['fexpert_succ_classes'] = fx['succ_classes']
v132['fexpert_r_value'] = fx['r_value']

joblib.dump(v132, 'submit/model/model_artifacts_v132.pkl')
print('v132 저장 완료 (v131 + F전문가 라우팅)')
