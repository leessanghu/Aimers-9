"""v106 실측(1077.4354648028)으로부터 D_true 역산.
먼저 v95 vs v106 아티팩트가 level_shift 외에 다른 게 없는지 검증(필수)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, joblib

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v106 = joblib.load('submit/model/model_artifacts_v106.pkl')

print('=== (0) 아티팩트 차이 검증 ===')
keys = sorted(set(v95) | set(v106))
diffs = []
for k in keys:
    if k not in v95:
        diffs.append((k, 'v95에 없음', v106[k]))
        continue
    if k not in v106:
        diffs.append((k, v95[k], 'v106에 없음'))
        continue
    a, b = v95[k], v106[k]
    same = (a is b)
    if not same:
        if isinstance(a, (int, float, str, bool, type(None))) and isinstance(b, (int, float, str, bool, type(None))):
            same = (a == b)
        elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            same = a.shape == b.shape and np.array_equal(a, b)
        else:
            same = True  # 동일 객체 참조가 아니면 여기선 판단보류(아래 주의문 참고)
    if not same:
        diffs.append((k, a, b))
if not diffs:
    print('  스칼라/배열 키에서 차이 없음 (level_shift 포함해서 아래 재확인)')
for k, a, b in diffs:
    print(f'  [{k}]  v95={a}   v106={b}')
print(f"  level_shift: v95={v95['level_shift']!r}  v106={v106['level_shift']!r}")
s = float(v106['level_shift']) - float(v95['level_shift'])
print(f'  -> 실제 적용된 추가 shift  s = {s:+.8f}')

print()
print('=== (1) D_true 역산 ===')
B = 0.249807
S0 = 1103.6568315036
S1 = 1077.4354648028
BS0 = B * (1 - S0 / 1e5)
BS1 = B * (1 - S1 / 1e5)
print(f'  BS0(v95)  = {BS0:.12f}')
print(f'  BS1(v106) = {BS1:.12f}')
dBS = BS1 - BS0
print(f'  ΔBS = {dBS:+.12f}   (점수차 {S1-S0:+.4f})')

# ΔBS = 2sD + s^2  ->  D = (ΔBS - s^2) / (2s)
D = (dBS - s * s) / (2 * s)
print(f'  => D_true = {D:+.8f}   (v95 예측평균 - 실제평균)')

print()
print('=== (2) 이게 뜻하는 것 ===')
opt = -D
gain = (1e5 / B) * D * D
print(f'  최적 추가 shift = {opt:+.8f}')
print(f'  그때 최대 이득  = {gain:+.4f} 점')
print(f'  (참고) fold A 추정 D=+0.007184 였다면 이득 = {(1e5/B)*0.007184**2:+.2f} 점')
print()
print(f'  v95의 level_shift({v95["level_shift"]:+.4f})를 빼면 raw 블렌드 편차 = {D - float(v95["level_shift"]):+.8f}')
