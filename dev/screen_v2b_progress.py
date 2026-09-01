"""스크리닝 v2b — 월 대신 '시즌 진행도(season progress)' 기반 가중.

왜 원시 game_month가 아니라 진행도인가:
  월 커버리지가 시즌마다 다르다. 3월은 2019/2024에만 있고 2020은 코로나로 5월 시작.
    2019: 3~10월 / 2020: 5~10월 / 2021~2023: 4~10월 / 2024: 3~10월
  -> 같은 '4월'이 2020엔 존재조차 안 하고, 2021의 4월은 개막이지만 2019의 4월은 2번째 달.
  진행도(시즌내 row_num 백분위)로 정규화해야 시즌 간 비교가 성립한다.

진단 근거 (fold A 2024, v50 로컬구성):
  진행도  0-10%  score  872   예측sd 0.0504
         30-40% score 1220   <- 최고
         70-80% score  768
         90-100% score 450   예측sd 0.0436  <- 붕괴
  in-season 투구수 분위로는 뚜렷한 패턴 없음 -> 원인변수는 '진행도'지 '표본성숙도'가 아님.
  즉 시즌 후반부는 (a) 테스트에 없을 가능성이 높고 (b) 본질적으로 예측이 가장 안 되는
  구간이다. 전체가중으로 학습하면 이 노이즈 구간이 트리 분할을 잠식한다.

주의: 이 계열은 '학습절차 변경'류(실측 0/6). 게이트 통과해도 실측 우선순위는 낮게.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "screen_v2b_cache"
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
month = pd.read_csv("../data/train.csv", encoding="utf-8-sig", usecols=["season", "game_month"])
assert len(month) == len(meta) and (month["season"].to_numpy() == meta["season"].to_numpy()).all()
mo = month["game_month"].to_numpy()
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
prog = meta.groupby("season")["row_num"].rank(pct=True).to_numpy(np.float64)
log(f"  진행도 range=[{prog.min():.3f},{prog.max():.3f}]")

# 테스트(2025 3~7월) 추정 진행도 구간: 2024에서 3~7월이 차지하는 진행도 범위로 근사
m24 = seasons == 2024
p37 = prog[m24 & (mo >= 3) & (mo <= 7)]
LO, HI = float(p37.min()), float(p37.max())
log(f"  2024 3~7월 = 진행도 {LO:.3f}~{HI:.3f} (테스트 추정구간)")


def recency(s_arr, ref, half_life=2.0):
    return 0.5 ** ((ref - s_arr) / half_life)


def w_none(bw, pr):
    return bw


# v2(원시 월) 결과: 시즌말을 '깎는' 방향은 fold A에서 -23~-57로 강하게 기각됨.
# -> 데이터를 빼는 방향은 죽었다. 남은 건 '더하는' 방향(개막부 상향)과 아주 완만한 축소뿐.
def w_prog_soft08(bw, pr):
    """테스트 추정구간 밖 0.8배 — 축소방향 중 가장 완만한 것만 잔존 확인"""
    return bw * np.where(pr > HI, 0.8, 1.0)


def w_prog_early15(bw, pr):
    """개막부(진행도<=0.15) 1.5배 — 데이터를 빼지 않고 더하는 방향"""
    return bw * np.where(pr <= 0.15, 1.5, 1.0)


def w_prog_early3x(bw, pr):
    """개막부 3배. 3월은 2019/2024에만 있어 학습표본 1.7%뿐인데
    테스트 공식샘플 5행 중 3행이 3월이었음 -> 희소성 보정 강판"""
    return bw * np.where(pr <= 0.15, 3.0, 1.0)


CANDIDATES = [
    ("baseline", w_none, "기준"),
    ("prog_soft08", w_prog_soft08, "절차변경"),
    ("prog_early15", w_prog_early15, "절차변경"),
    ("prog_early3x", w_prog_early3x, "절차변경"),
]

FOLDS = [(2023, 2024, "A"), (2021, 2022, "C")]
results = {}
for upto, val, tag in FOLDS:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv_all = y[va_m]
    mv = mo[va_m]
    bw = recency(seasons, upto)
    Xtr, ytr, Xva = X.loc[tr_m], y[tr_m], X.loc[va_m]
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
                preds.append(np.load(f)); continue
            ts = time.time()
            w = wfn(bw, prog)[tr_m]
            keep = w > 0
            m_ = HistGradientBoostingClassifier(**BASE_HGB, random_state=seed)
            m_.fit(Xtr.loc[keep], ytr[keep], sample_weight=w[keep])
            p = m_.predict_proba(Xva)[:, 1]
            np.save(f, p); preds.append(p)
            log(f"    [{tag}/{name}/s{seed}] n={keep.sum():,} iters={m_.n_iter_} ({time.time()-ts:.0f}s)")
        avg = np.mean(preds, axis=0)
        row = {}
        for segnm, seg in SEGS:
            row[segnm] = seg_score(avg, seg)
            row[segnm + "_폭"] = max(seg_score(p, seg) for p in preds) - min(seg_score(p, seg) for p in preds)
        results[(tag, name)] = row
        log(f"  {tag}/{name:<13} 3-7월={row['3-7월(주판정)']:8.2f}(폭{row['3-7월(주판정)_폭']:.2f}) "
           f"전체={row['전체(보조)']:8.2f}")

print()
print("=" * 92)
print("스크리닝 v2b (시즌 진행도 가중)  — delta는 baseline 대비")
print("=" * 92)
print(f"{'fold':<5}{'후보':<14}{'3-7월Δ':>10}{'시드폭':>8}{'전체Δ':>10}  판정")
for upto, val, tag in FOLDS:
    b = results.get((tag, "baseline"))
    if b is None:
        continue
    for name, wfn, kind in CANDIDATES:
        r = results.get((tag, name))
        if r is None or name == "baseline":
            continue
        d1 = r["3-7월(주판정)"] - b["3-7월(주판정)"]
        d2 = r["전체(보조)"] - b["전체(보조)"]
        sp = r["3-7월(주판정)_폭"]
        v = "통과(절차변경:우선순위↓)" if (d1 > 3 and d2 > -3 and d1 > sp) else "기각"
        print(f"{tag:<5}{name:<14}{d1:+10.2f}{sp:8.2f}{d2:+10.2f}  {v}")
log(f"총 {time.time()-t0:.0f}s")
