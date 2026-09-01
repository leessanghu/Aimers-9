"""v98 = v95 + 블렌드 가중치 재배분 (재학습 불필요, 가중치 값만 변경).
mc5를 hurdle급(0.190)으로 상향, 원래 작은 비중이던 multires/condball/countresid에서 차감.
base는 그대로 유지(별도 실측 대상인 mc5 상향 효과만 분리해서 보기 위함 - v97과 따로 검증).
"""
import joblib

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v98 = dict(v95)

print('=== 기존 가중치 ===')
keys = ['base_weight', 'hurdle_weight', 'multires_weight', 'ordinal_weight', 'midother_weight',
        'condball_weight', 'countresid_weight', 'future50_weight', 'mc5_weight', 'ingame_weight']
for k in keys:
    print(f'  {k:20s} {v95[k]:.4f}')

v98['base_weight'] -= 0.02      # 실측(v50,v58)으로 이미 검증된 방향: base 낮을수록 좋았음
v98['multires_weight'] -= 0.015  # 원래 가장 작은 비중(4.8%) - 약한 축
v98['condball_weight'] -= 0.0085  # idea54에서 추가된 축, 실측 기여 미미(80개 이력상 '나머지+1.79'에 포함)
v98['countresid_weight'] -= 0.0085
v98['mc5_weight'] += 0.052       # hurdle급(19%)으로 상향 - 오늘 유일하게 실측 검증된 강한 축

print('\n=== 신규 가중치 ===')
tot = 0.0
for k in keys:
    print(f'  {k:20s} {v98[k]:.4f}')
    tot += v98[k]
print(f'합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v98, 'submit/model/model_artifacts_v98.pkl')
print('\nv98 저장 완료')
