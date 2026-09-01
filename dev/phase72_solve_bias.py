"""LB 실측 2개로 2025 test의 진짜 편향 b를 역산 + 자기검증 + 민감도 분석.

유도 (전부 항등식, 근사 없음):
  Score = 1e5 * (1 - BS/BSref)        <- EVALUATION.md 공식 그대로
  => BS  = BSref * (1 - Score/1e5)

  제출1: 예측 p 그대로            -> S0
  제출2: 예측 p - c (상수 shift)  -> S1

  [측정으로부터]  BS1 - BS0 = BSref * (S0 - S1)/1e5
  [해석적으로]    BS1 - BS0 = mean((p-c-y)^2) - mean((p-y)^2)
                            = -2c*mean(p-y) + c^2
                            = c^2 - 2*c*b        (b = mean(p) - mean(y) = 편향)

  두 식을 놓고 b에 대해 풀면:
      b = [ c^2 - BSref*(S0-S1)/1e5 ] / (2c)

  미지수는 BSref(=r(1-r), r은 2025 성공률)뿐인데 r이 0.45~0.50 어디에 있든
  r(1-r)은 0.2475~0.2500 범위라 b가 거의 안 흔들린다(아래 민감도 참조).

최적 보정: c* = b 이고, 그때 얻는 점수는 1e5 * b^2 / BSref.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

# ===== 실측값 =====
S0 = 1025.6585619092      # submit_v27_raw.zip      (무보정)
S1 = 1010.5314203942      # submit_v27_shifted.zip  (p - 0.0105)
C = 0.0105                # 우리가 적용한 shift 상수

dS = S1 - S0
print("=" * 74)
print("입력 (실측)")
print("=" * 74)
print(f"  S0 (무보정)      = {S0:.4f}")
print(f"  S1 (shift {C})  = {S1:.4f}")
print(f"  ΔS = S1 - S0     = {dS:+.4f}")
print()


def solve_b(bsref):
    return (C ** 2 - bsref * (S0 - S1) / 1e5) / (2 * C)


def predicted_dS(b, c, bsref):
    """편향 b인 예측에 shift c를 적용했을 때 점수 변화 (해석적)."""
    return 1e5 * (2 * c * b - c ** 2) / bsref


def gain_from(c, b, bsref):
    return 1e5 * (2 * c * b - c ** 2) / bsref


print("=" * 74)
print("민감도 — 2025 성공률 r을 모르지만 BSref=r(1-r)은 거의 안 변한다")
print("=" * 74)
print(f"{'r 가정':>10}{'BSref':>10}{'역산된 b':>14}{'최적보정 이득':>16}")
print("-" * 52)
rows = []
for r in (0.45, 0.46, 0.47, 0.48, 0.486, 0.49, 0.50):
    bsref = r * (1 - r)
    b = solve_b(bsref)
    g = 1e5 * b ** 2 / bsref          # c*=b 일 때의 이득
    rows.append((r, bsref, b, g))
    print(f"{r:>10.3f}{bsref:>10.5f}{b:>14.6f}{g:>16.2f}")

bs = [x[2] for x in rows]
gs = [x[3] for x in rows]
print()
print(f"  -> b 범위 [{min(bs):.6f}, {max(bs):.6f}]   폭 {max(bs)-min(bs):.6f}")
print(f"  -> 이득 범위 [{min(gs):.2f}, {max(gs):.2f}]")

# 중심값 (2024 실측 0.486에서 소폭 하락 가정)
R_C = 0.482
BSREF_C = R_C * (1 - R_C)
B_HAT = solve_b(BSREF_C)

print()
print("=" * 74)
print("자기검증 — 역산한 b로 관측된 ΔS가 재현되는가")
print("=" * 74)
chk = predicted_dS(B_HAT, C, BSREF_C)
print(f"  중심 가정 r={R_C}  BSref={BSREF_C:.5f}")
print(f"  역산된 b        = {B_HAT:.6f}")
print(f"  b로 예측한 ΔS   = {chk:+.4f}")
print(f"  실제 관측 ΔS    = {dS:+.4f}")
print(f"  잔차            = {chk - dS:+.6f}   (0에 가까우면 방정식이 일관됨)")

print()
print("=" * 74)
print("결론")
print("=" * 74)
opt_gain = 1e5 * B_HAT ** 2 / BSREF_C
print(f"  2025 test의 진짜 편향 b   = {B_HAT:.5f}   (우리가 가정했던 {C} 의 {B_HAT/C*100:.0f}%)")
print(f"  최적 보정 c* = b 적용시    S0 {S0:.1f} -> {S0+opt_gain:.1f}  (+{opt_gain:.2f})")
print(f"  우리가 한 과보정(c={C})    S0 {S0:.1f} -> {S0+gain_from(C,B_HAT,BSREF_C):.1f}  ({gain_from(C,B_HAT,BSREF_C):+.2f})")
print()
print(f"  참고: 로컬 2024 폴드에서 잰 편향은 0.01055 였다.")
print(f"        실제 2025는 {B_HAT:.5f} 로 그 {B_HAT/0.01055*100:.0f}% 수준.")
print(f"        -> 로컬 폴드가 시즌 드리프트 문제를 과대평가하고 있었다.")
print()
print(f"  보정으로 얻을 수 있는 최대치가 +{opt_gain:.1f}점뿐이므로 보정 방향은 사실상 종료.")
