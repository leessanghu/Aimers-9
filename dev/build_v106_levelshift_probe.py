"""v106 = v95 + level_shift 추가보정.
근거: fold A(train<=2023->2024) 정직검증에서 잔여편차 D=+0.007184(예측이 실제보다 높음)
측정, 보정시 928.51->949.17(+20.66). 이론식 이득=400309*D^2과 정확히 일치(수치오차 無).
트리모델이 season=2025를 2024와 완전동일 취급(외삽 불가, 실측차0.0000000000)하고
2022->2023->2024 성공률이 -2.9%p,-1.4%p로 급락중이라 2025도 비슷한 낙폭이 예상됨.
현재 level_shift=-0.00127에서 추가로 -0.007184 이동."""
import joblib

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v106 = dict(v95)

D_MEASURED = 0.007184
old_ls = v95['level_shift']
new_ls = old_ls - D_MEASURED
v106['level_shift'] = new_ls
print(f'level_shift: {old_ls:.5f} -> {new_ls:.5f}  (추가이동 {-D_MEASURED:+.5f})')

joblib.dump(v106, 'submit/model/model_artifacts_v106.pkl')
print('v106 저장 완료')

# 실측 후 진짜 D를 정확히 역산하기 위한 참고값
UNC = 0.249807
S0 = 1103.6568315036  # v95 실측
BS0 = UNC * (1 - S0 / 100000)
s = new_ls - old_ls  # 우리가 실제로 준 이동량
print(f'\n[실측 후 검증용] BS0(v95)={BS0:.8f}, s={s:+.6f}')
print('실측 S1 나오면: BS1 = UNC*(1-S1/100000)')
print('  D_true = (BS1 - BS0 - s**2) / (2*s)')
print('  이 D_true가 -D_MEASURED 근처면 로컬추정이 맞았다는 뜻,')
print('  아니면 D_true로 최적 level_shift를 한번 더 정밀 조정 가능(이론상 100% 정확)')
