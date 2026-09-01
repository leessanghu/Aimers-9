"""사후보정 — idea16 최적조합(base=0.3,hur=0.4,mr=0.1,or=0.2)에 isotonic regression을
적용해 실제 Brier 개선이 있는지 확인. fold 내부를 시간순으로 절반 나눠 앞쪽에서 보정곡선을
학습하고 뒤쪽에서 평가(미래 데이터로 과거 보정 적용 -- 실전과 동일 구조).
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

CD = "idea13_cache"
t0 = time.time()
W = (0.30, 0.40, 0.10, 0.20)  # base, hur, mr, or


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
row_num = meta["row_num"].to_numpy(np.float64)

for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)

    def sc(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    p_mr = np.mean([np.load(f"{CD}/{tag}_multires_s{s}.npy") for s in [42, 7]], axis=0)
    p_or = np.mean([np.load(f"{CD}/{tag}_ordinal_s{s}.npy") for s in [42, 7]], axis=0)
    blend = W[0] * base3 + W[1] * hur + W[2] * p_mr + W[3] * p_or
    blend = np.clip(blend, 0, 1)

    row_va = row_num[va_m]
    order = np.argsort(row_va)
    half = len(order) // 2
    fit_idx, eval_idx = order[:half], order[half:]

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(blend[fit_idx], yv[fit_idx])
    p_cal_eval = iso.predict(blend[eval_idx])

    raw_score_full = sc(blend)
    raw_score_eval = sc(blend[eval_idx]) if False else (
        1e5 * (1 - np.mean((blend[eval_idx] - yv[eval_idx]) ** 2) / (yv[eval_idx].mean() * (1 - yv[eval_idx].mean())))
    )
    cal_score_eval = (
        1e5 * (1 - np.mean((p_cal_eval - yv[eval_idx]) ** 2) / (yv[eval_idx].mean() * (1 - yv[eval_idx].mean())))
    )
    log(f"fold {tag}: 전체원본={raw_score_full:.2f}  |  평가절반: 원본={raw_score_eval:.2f}  보정후={cal_score_eval:.2f}  "
       f"(보정효과 {cal_score_eval-raw_score_eval:+.2f})")
    log(f"  예측평균: 원본={blend[eval_idx].mean():.4f}  보정후={p_cal_eval.mean():.4f}  실제={yv[eval_idx].mean():.4f}")

log(f"총 {time.time()-t0:.0f}s")
