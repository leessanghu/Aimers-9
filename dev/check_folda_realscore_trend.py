"""fold A 로컬(H1/H2 평균) vs 실측Δ 세 점을 선형회귀로 외삽.
데이터점 3개뿐이라 자유도 1인 극도로 약한 적합이지만, 방향성 정도는 볼 수 있다."""
import numpy as np

# (fold A H1/H2 평균, 실측 v95 대비 Δ)
x = np.array([0.90, 1.95, 8.42])   # v108, v107, v104
y = np.array([-1.19, -8.78, -35.67])
names = ['v108(XGB)', 'v107(physhead)', 'v104(perhead iso)']

n = len(x)
xbar, ybar = x.mean(), y.mean()
b = np.sum((x - xbar) * (y - ybar)) / np.sum((x - xbar) ** 2)
a = ybar - b * xbar
resid = y - (a + b * x)
print(f'적합: 실측Δ = {a:.2f} + {b:.2f} * foldA   (n=3, 자유도=1)')
print(f'잔차: {resid}   (자유도1이라 신뢰구간 사실상 무의미, 참고용)')

for nm, xv in [('foldA=0', 0.0), ('구종축A(−0.85)', -0.85), ('mc6원본(−2.28)', -2.28)]:
    pred = a + b * xv
    print(f'  {nm:<18} 외삽 예측실측Δ = {pred:+.2f}')

print('\n[주의] n=3, df=1인 회귀는 사실상 두 구간 기울기의 평균일 뿐이다.')
print(' 기울기가 진짜인지, 우연히 세 점이 정렬된 것인지 구분 불가능.')
print(' 그래도 "fold A가 클수록 실측이 더 나쁘다"는 부호는 3쌍 모두 일관됐다는')
print(' 사실 자체는 유효하다 (단조성은 재현됨, 정확한 기울기값은 아님).')
