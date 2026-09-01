"""아이디어38 — '확대엔트리/콜업' 축. 사용자 관찰에서 출발.

관찰(확인됨): 2024 시즌 마지막 20% 구간에서 등판 고유투수가 245~269명 -> 295명(+16%).
평균경력은 오히려 최고(4343)이고 신인비율은 낮음 -> 신인 유입이 아니라
'평소 안 나오던 선수들의 대거 등판'(9월 확대엔트리, 엔트리 28->33).
그 구간에서 모델 점수가 878 -> 549로 붕괴.

game_month는 이미 피처라 "9월이다"는 모델이 안다. 없는 것은 **투수 개인 수준**:
  - 이 투수가 이번 시즌 언제 처음 등판했나 (늦게 콜업된 선수인가)
  - 이 투수가 이번 시즌 얼마나 드문드문 나오나 (등판 밀도)
  - 지금이 그 투수의 시즌 등판 이력 중 어디쯤인가

추정량은 partial_gain(잔차 편상관, 모델학습 없음 -> 시드노이즈 0, sigma~4.0점).
스크리닝 v2 규칙상 이 계열은 '새 정보 추가'류(실측 4/4 성공).
leakage: 각 행은 자기 투수의 '그 행 이전' 기록만 사용(expanding). 미래 참조 없음.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

t0 = time.time()
N_REF = 253507
SIGMA = 1e5 * (1.0 / N_REF)  # partial_gain 스케일의 1시그마 근사


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def partial_gain(y, p, z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    if z.std() == 0:
        return 0.0, 0.0
    A = np.column_stack([np.ones(len(y)), p])
    cy = np.linalg.lstsq(A, y, rcond=None)[0]
    cz = np.linalg.lstsq(A, z, rcond=None)[0]
    ry, rz = y - A @ cy, z - A @ cz
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


log("로드...")
meta = pd.read_parquet("featcache_meta.parquet")
raw = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                  usecols=["game_month", "game_type", "pitcher_id", "season", "row_id"])
meta["mo"] = raw["game_month"].to_numpy()
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

# 시즌 진행도(시즌내 row_num 백분위) — 시즌마다 월 커버리지가 달라 정규화 필수
meta["prog"] = meta.groupby("season")["row_num"].rank(pct=True)
log("콜업 피처 구성 (전부 expanding, 미래참조 없음)...")

srt = meta.sort_values(["pitcher_id", "season", "row_num"])
gp = srt.groupby(["pitcher_id", "season"], sort=False)

# 1) 그 투수가 이번 시즌 처음 등장한 시점의 진행도 (시즌 내내 상수 -> 콜업 시기)
first_prog = gp["prog"].transform("min")
# 2) 지금이 그 투수의 이번시즌 등판 이력 중 몇번째 투구인지 (expanding count)
pitch_idx = gp.cumcount().astype(np.float64)
# 3) 콜업 이후 얼마나 시간이 흘렀나
since_callup = srt["prog"].to_numpy() - first_prog.to_numpy()
# 4) 등판 밀도: 시즌 내 투구수 / 경과 진행도 (드문드문 나오는 투수 = 낮음)
density = pitch_idx.to_numpy() / np.maximum(since_callup, 1e-3)
# 5) 늦은 콜업 플래그 x 현재 진행도 (확대엔트리 상호작용)
late_callup = (first_prog.to_numpy() > 0.5).astype(np.float64)

feats = pd.DataFrame(index=srt.index)
feats["cu_first_prog"] = first_prog.to_numpy()
feats["cu_pitch_idx"] = pitch_idx.to_numpy()
feats["cu_since_callup"] = since_callup
feats["cu_density"] = np.log1p(density)
feats["cu_late_flag"] = late_callup
feats["cu_late_x_prog"] = late_callup * srt["prog"].to_numpy()
feats["cu_lowdensity_x_prog"] = (density < np.nanmedian(density)).astype(np.float64) * srt["prog"].to_numpy()
feats = feats.sort_index()
log(f"  {list(feats.columns)}")

# fold A 검증행(2024)에서 partial_gain
va = seasons == 2024
b = np.mean([np.load(f"phase90_cache/A_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
h = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) * np.load(f"phase90_cache/A_snc_{n}.npy")
             for n in ["d6", "d8"]], axis=0)
m = np.mean([np.load(f"idea13_cache/A_multires_s{k}.npy") for k in [42, 7]], axis=0)
o = np.mean([np.load(f"idea13_cache/A_ordinal_s{k}.npy") for k in [42, 7]], axis=0)
d = np.mean([np.load(f"idea31_cache/A_midaxis_s{k}.npy") for k in [42, 7]], axis=0)
p_base = np.clip(0.20 * b + 0.40 * h + 0.10 * m + 0.20 * o + 0.10 * d, 1e-6, 1 - 1e-6)

yv = y[va]
mv = meta["mo"].to_numpy()[va]
seg37 = (mv >= 3) & (mv <= 7)

print()
print("=" * 78)
print(f"{'후보':<24}{'전체2024':>12}{'3-7월(주판정)':>16}{'부호':>8}")
print(f"(1시그마 ~ {SIGMA:.2f}점, 통과기준 4시그마={4*SIGMA:.1f})")
print("-" * 78)
rows = []
for c in feats.columns:
    z = feats[c].to_numpy()[va]
    g_all, pc_all = partial_gain(yv, p_base, z)
    g_37, pc_37 = partial_gain(yv[seg37], p_base[seg37], z[seg37])
    rows.append((c, g_all, g_37, pc_37))
    print(f"{c:<24}{g_all:12.2f}{g_37:16.2f}{('+' if pc_37>0 else '-'):>8}")
print()
best = max(rows, key=lambda r: r[2])
print(f"최고: {best[0]}  3-7월 partial_gain={best[2]:.2f}")
print("참고: 과거 채택 피처 bat_inseason_smooth=+17.05(6.6시그마), 기각선 4시그마")
log(f"총 {time.time()-t0:.0f}s")
