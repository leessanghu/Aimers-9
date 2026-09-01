"""v99 = v95 + 약한 헤드(multires/condball/countresid) 완전 제거, mc5로 전량 이전.
base는 그대로 유지(v98에서 이미 별도로 테스트하므로 여기선 순수 '헤드 제거+mc5 집중' 효과만 분리)."""
import joblib

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v99 = dict(v95)

keys = ['base_weight', 'hurdle_weight', 'multires_weight', 'ordinal_weight', 'midother_weight',
        'condball_weight', 'countresid_weight', 'future50_weight', 'mc5_weight', 'ingame_weight']
print('=== 기존 가중치 ===')
for k in keys:
    print(f'  {k:20s} {v95[k]:.4f}')

freed = v99['multires_weight'] + v99['condball_weight'] + v99['countresid_weight']
v99['multires_weight'] = 0.0
v99['condball_weight'] = 0.0
v99['countresid_weight'] = 0.0
v99['mc5_weight'] += freed

print(f'\n제거된 가중치 합 = {freed:.4f}  -> mc5로 이전')
print('\n=== 신규 가중치 ===')
tot = 0.0
for k in keys:
    print(f'  {k:20s} {v99[k]:.4f}')
    tot += v99[k]
print(f'합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v99, 'submit/model/model_artifacts_v99.pkl')
print('\nv99 저장 완료')
