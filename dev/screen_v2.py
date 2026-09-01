"""스크리닝 프로토콜 v2 — 기존 스크리닝의 3대 결함을 고친 공용 하네스.

기존 결함:
  (1) 검증셋 = 2024 전체. 그런데 테스트는 2025 시즌 전반부(3~7월)로 추정됨
      (공식샘플 5행 중 3행이 3월, 3월은 학습의 1.7% -> p~5e-5).
      fold A -> LB 스케일계수 1.143도 3-7월 구간(1.128)에서만 재현됨.
      -> 전체2024로 재면 테스트에 없을 수 있는 8-10월(30%)이 판정을 오염시킴.
  (2) 단일/2시드로 판정. 세션 확립 노이즈 하한: 최종블렌드 delta ±2.
  (3) "새 정보 추가"와 "학습절차 변경"을 구분 안 함.
      실측 전적: 새 정보 추가 4/4 성공(multires/ordinal/가중치/midaxis),
                 학습절차 변경 0/6 실패(refit-closure 전멸).

v2 규칙:
  주판정 = fold A의 2024년 3~7월 (테스트 분포 매칭)
  보조   = fold A 전체2024, fold C 3~7월
  시드   = 3시드, 시드폭도 함께 보고
  게이트 = 주판정 delta > +3  AND  3시드 부호 일치  AND  보조가 -3 미만으로 안 깨짐
  분류   = 절차변경류는 게이트 통과해도 실측 우선순위 하향(0/6 전적)

사용법: SCREEN_CANDIDATES 에 (이름, weight_fn) 추가. weight_fn(meta, month, seasons,
train_upto) -> sample_weight 배열. None이면 기존 recency_weight만.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "screen_v2_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7, 2024]

BASE_HGB = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20, max_depth=6, max_leaf_nodes=31)


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
month = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                    usecols=["season", "game_month"])
assert len(month) == len(meta) and (month["season"].to_numpy() == meta["season"].to_numpy()).all()
mo = month["game_month"].to_numpy()
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
log(f"  X={X.shape}  월범위={mo.min()}~{mo.max()}")


def recency(seasons_arr, ref, half_life=2.0):
    return 0.5 ** ((ref - seasons_arr) / half_life)


# ---------------- 후보 정의 ----------------
# w_fn(base_w, mo, seasons, upto) -> 최종 sample_weight
def w_none(base_w, m, s, upto):
    return base_w


def w_month_soft07(base_w, m, s, upto):
    """시즌말(8-10월) 0.7배 — 테스트(3-7월) 분포로 완만히 이동"""
    return base_w * np.where(m >= 8, 0.7, 1.0)


def w_month_soft05(base_w, m, s, upto):
    return base_w * np.where(m >= 8, 0.5, 1.0)


def w_month_gauss(base_w, m, s, upto):
    """테스트 중심(5월) 가우시안 커널, sigma=2.5"""
    return base_w * np.exp(-((m - 5.0) ** 2) / (2 * 2.5 ** 2))


CANDIDATES = [
    ("baseline", w_none, "기준"),
    ("mo_soft07", w_month_soft07, "절차변경"),
    ("mo_soft05", w_month_soft05, "절차변경"),
    ("mo_gauss", w_month_gauss, "절차변경"),
]

FOLDS = [(2023, 2024, "A"), (2021, 2022, "C")]
results = {}

for upto, val, tag in FOLDS:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv_all = y[va_m]
    mv = mo[va_m]
    base_w = recency(seasons, upto)
    Xtr = X.loc[tr_m]
    ytr = y[tr_m]
    Xva = X.loc[va_m]

    SEGS = [("3-7월(주판정)", (mv >= 3) & (mv <= 7)), ("전체(보조)", np.ones(len(mv), bool))]

    def seg_score(p, seg):
        yy = yv_all[seg]
        r = yy.mean(); BS = r * (1 - r)
        return 1e5 * (1 - np.mean((np.clip(p[seg], 0, 1) - yy) ** 2) / BS)

    for name, wfn, kind in CANDIDATES:
        preds = []
        for seed in SEEDS:
            f = f"{CD}/{tag}_{name}_s{seed}.npy"
            if os.path.exists(f):
                preds.append(np.load(f))
                continue
            ts = time.time()
            w = wfn(base_w, mo, seasons, upto)[tr_m]
            m_ = HistGradientBoostingClassifier(**BASE_HGB, random_state=seed)
            m_.fit(Xtr, ytr, sample_weight=w)
            p = m_.predict_proba(Xva)[:, 1]
            np.save(f, p)
            preds.append(p)
            log(f"    [{tag}/{name}/s{seed}] iters={m_.n_iter_} ({time.time()-ts:.0f}s)")
        avg = np.mean(preds, axis=0)
        row = {}
        for segnm, seg in SEGS:
            row[segnm] = seg_score(avg, seg)
            row[segnm + "_시드폭"] = max(seg_score(p, seg) for p in preds) - min(seg_score(p, seg) for p in preds)
        results[(tag, name)] = row
        log(f"  {tag}/{name:<12} 3-7월={row['3-7월(주판정)']:8.2f}(폭{row['3-7월(주판정)_시드폭']:.2f})  "
           f"전체={row['전체(보조)']:8.2f}(폭{row['전체(보조)_시드폭']:.2f})")

print()
print("=" * 92)
print("스크리닝 v2 결과  (delta는 baseline 대비)")
print("=" * 92)
print(f"{'fold':<5}{'후보':<13}{'분류':<8}{'3-7월Δ':>10}{'시드폭':>8}{'전체Δ':>10}{'시드폭':>8}  판정")
for upto, val, tag in FOLDS:
    b = results.get((tag, "baseline"))
    if b is None:
        continue
    for name, wfn, kind in CANDIDATES:
        r = results.get((tag, name))
        if r is None:
            continue
        d1 = r["3-7월(주판정)"] - b["3-7월(주판정)"]
        d2 = r["전체(보조)"] - b["전체(보조)"]
        sp = r["3-7월(주판정)_시드폭"]
        if name == "baseline":
            verdict = "-"
        elif d1 > 3 and d2 > -3 and d1 > sp:
            verdict = "통과" + ("(절차변경:우선순위↓)" if kind == "절차변경" else "")
        else:
            verdict = "기각"
        print(f"{tag:<5}{name:<13}{kind:<8}{d1:+10.2f}{sp:8.2f}{d2:+10.2f}{r['전체(보조)_시드폭']:8.2f}  {verdict}")
log(f"총 {time.time()-t0:.0f}s")
