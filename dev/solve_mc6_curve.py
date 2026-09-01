"""실측 2점으로 mc6 가중치곡선의 A, V를 정확히 연립해서 푼다.

ΔScore(s) = -K*(2sA + s^2 V)
  s1=0.03 -> +1.1775 (v112)
  s2=0.10 -> +3.6309 (v114)
두 방정식 두 미지수 -> A, V 유일해. 더 이상 V를 fold A로 근사하지 않아도 된다.

+ 측정오차 민감도: 각 실측에 SE만큼 오차가 있을 때 s*와 최대이득이 얼마나 흔들리는지.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

B = 0.249807
K = 1e5 / B
BASE = 1103.6568315036
S1, R1 = 0.03, 1104.8342852052
S2, R2 = 0.10, 1107.2877112561
D1, D2 = R1 - BASE, R2 - BASE
print(f'v112: s={S1}  ΔScore={D1:+.4f}')
print(f'v114: s={S2}  ΔScore={D2:+.4f}\n')


def solve(d1, d2, s1=S1, s2=S2):
    # 2sA + s^2 V = -Δ/K
    b1, b2 = -d1 / K, -d2 / K
    M = np.array([[2 * s1, s1 ** 2], [2 * s2, s2 ** 2]])
    A, V = np.linalg.solve(M, np.array([b1, b2]))
    return A, V


A, V = solve(D1, D2)
s_star = -A / V
gain_max = K * A ** 2 / V
print(f'=== 정확해 ===')
print(f'  A = {A:+.4e}   V = {V:.4e}')
print(f'  s* = {s_star:+.4f}   최대이득 = {gain_max:+.2f}점')
print(f'  (이전 fold A 근사: V=1.49e-04 -> s*=0.34, 최대이득 +7.05)')
print(f'  실제 V가 {V/1.49e-4:.2f}배 작아서 최적점이 더 뒤에 있었음\n')

print(f'=== 검증 (2점이 정확히 재현되는지) ===')
for s, d in [(S1, D1), (S2, D2)]:
    pred = -K * (2 * s * A + s ** 2 * V)
    print(f'  s={s}: 예측={pred:+.4f}  실제={d:+.4f}  오차={abs(pred-d):.2e}')

print(f'\n=== 가중치별 예상 ΔScore ===')
print(f'{"s":>7}{"예상 ΔScore":>14}{"최대대비":>10}')
for s in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.48, 0.50, 0.55, 0.60):
    g = -K * (2 * s * A + s ** 2 * V)
    print(f'{s:>7.2f}{g:>14.2f}{g/gain_max*100:>9.1f}%')

print(f'\n=== 측정오차 민감도 ===')
print('  (SE: s=0.03에서 ~0.40, s=0.10에서 ~1.33 로 추정. 두 점을 ±SE 흔들어봄)')
SE1, SE2 = 0.40, 1.33
rows = []
for e1 in (-SE1, 0, SE1):
    for e2 in (-SE2, 0, SE2):
        try:
            a2, v2 = solve(D1 + e1, D2 + e2)
            if v2 <= 0:
                continue
            rows.append((-a2 / v2, K * a2 ** 2 / v2))
        except np.linalg.LinAlgError:
            continue
ss = np.array([r[0] for r in rows]); gg = np.array([r[1] for r in rows])
print(f'  s* 범위: {ss.min():+.3f} ~ {ss.max():+.3f}  (중앙 {s_star:+.3f})')
print(f'  최대이득 범위: {gg.min():+.2f} ~ {gg.max():+.2f}점  (중앙 {gain_max:+.2f})')

print(f'\n=== 보수적 추천: 최악(s*가 가장 작은) 시나리오에서도 이득인 s ===')
s_worst = ss.min()
for s in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40):
    worst_gain = min(-K * (2 * s * solve(D1 + e1, D2 + e2)[0]
                           + s ** 2 * solve(D1 + e1, D2 + e2)[1])
                     for e1 in (-SE1, 0, SE1) for e2 in (-SE2, 0, SE2)
                     if solve(D1 + e1, D2 + e2)[1] > 0)
    best_gain = -K * (2 * s * A + s ** 2 * V)
    print(f'  s={s:.2f}: 중앙예측 {best_gain:+6.2f}   최악시나리오 {worst_gain:+6.2f}')
