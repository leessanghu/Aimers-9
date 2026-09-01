"""헤드룸 분해 — 2024 폴드에서 '정답을 아는' 상한들과 우리 모델을 직접 비교.

각 상한은 2024 정답을 컨닝해서 만든 것(실전 불가). 우리가 어디서 얼마를 잃는지 특정용.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np, pandas as pd
from metrics import evaluate

df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
va = df[df.season==2024].copy()
y = va.control_success.to_numpy()
r = y.mean()
print(f"2024 valid n={len(va):,}  실제 성공률={r:.4f}  baseline_var={r*(1-r):.4f}\n")

def show(p, tag):
    p = np.clip(p, 1e-6, 1-1e-6)
    b = evaluate(y, p)["bss"]
    print(f"  {tag:44s} score={max(0,b*1e5):8.1f}")
    return b

print("=== 우리 모델 (phase27 측정치) ===")
print(f"  {'v12 blend / cat단독+작년':44s} score=   805.8 / 813.9\n")

print("=== 컨닝 상한 (2024 정답 사용) ===")
show(np.full(len(va), r), "상수(리그평균)")

# 1) 투수 정체성만: 그 투수의 2024 실제 성공률
pit = va.groupby("pitcher_id").control_success.transform("mean")
show(pit.to_numpy(), "투수의 2024 실제 성공률 (표본노이즈 포함)")

# 노이즈 제거판: 축소 추정 (경험적 베이즈)
g = va.groupby("pitcher_id").control_success.agg(["sum","count"])
var_true = 0.0555**2
k_opt = r*(1-r)/var_true
sm = (g["sum"] + k_opt*r)/(g["count"] + k_opt)
show(va.pitcher_id.map(sm).to_numpy(), f"투수 2024 성공률 축소판 (K={k_opt:.0f})")

# 2) 투수 + 볼카운트
va["cs"] = va.balls_before*4 + va.strikes_before
cnt = va.groupby("cs").control_success.transform("mean")
show((va.pitcher_id.map(sm).to_numpy() + cnt.to_numpy() - r), "투수축소 + 볼카운트 주효과")

# 3) 투수 + 볼카운트 + 이닝 + 주자 + 손잡이
for extra, nm in [(["cs"],"볼카운트"), (["cs","inning"],"+이닝"),
                  (["cs","inning","num_runners_on"],"+주자"),
                  (["cs","inning","num_runners_on","batter_hand"],"+타자손")]:
    eff = np.zeros(len(va))
    for c in extra:
        eff = eff + va.groupby(c).control_success.transform("mean").to_numpy() - r
    show(va.pitcher_id.map(sm).to_numpy() + eff, f"투수축소 + {nm} 주효과들")

# 4) 완전 컨닝: (투수, 볼카운트) 셀 평균
cell = va.groupby(["pitcher_id","cs"]).control_success.transform("mean")
show(cell.to_numpy(), "(투수 x 볼카운트) 셀 실제평균 [과적합 상한]")
