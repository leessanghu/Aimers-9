"""idea72 — 5-class softmax 분류로 문제를 재구성. 지금까지와 근본적으로 다른 귀납편향.

배경: PCA 진단상 모든 멤버가 PC1(95.1%)에 붕괴한 원인은 전부 같은 귀납편향
(GBDT + 이진타깃 회귀/MultiRMSE aux head)이기 때문이다. MLP로 편향을 바꾸려는
시도(idea61~71)는 성능이 안 나와 전부 실패했다. 이 실험은 **모델클래스는 CatBoost로
유지하되(성능 확보) 문제 정식화를 바꾼다**.

라벨 복원 전수감사로 확인된 결정론적 구조 (오차 0.00bp):
    middle     14.96%  성공률  0.000%
    reverse    19.49%  성공률  0.000%
    nd&ball    26.53%  성공률 59.100%
    nd&strike  27.93%  성공률 93.367%
    nd&기타     11.09%  성공률 95.769%
    -> 재구성 52.3733% == 실제 52.3733%

즉 5-class 확률만 맞히면 P(success)=Σ P(c)*E[y|c]로 결정된다. 기존 멤버는 전부
이진 회귀(각 head 독립, 확률합 제약 없음)인데, softmax는 5확률의 합=1이라는
구조적 제약을 걸어 학습한다 -- 같은 GBDT여도 최적화 표면이 다르므로 오차구조가
달라질 여지가 있다.

판정: (1) 단독 성능이 기존 멤버급(base 855)인가 (2) corr(v66)이 기존 aux축(0.98)보다
낮은가 (3) blend 손익분기 rho < sqrt(B1/B2)를 넘는가.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

CD = "idea72_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
season = meta["season"].to_numpy(np.float64)
cls = np.load("cls5_labels.npy")

UPTO, VAL = 2023, 2024
tr, va = season <= UPTO, season == VAL
yv = y[va]
r = float(yv.mean())
bs = r * (1 - r)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / bs)

avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
base = avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6", "d8", "sub")])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ("d6", "d8")], axis=0)
mr = avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42, 7)])
od = avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42, 7)])
mo = avg([f"idea46_cache/A_midother_s{s}.npy" for s in (42, 7)])
cb = avg([f"idea54_cache/A_cond_ball_s{s}.npy" for s in (42, 7)])
cr = avg([f"idea54_cache/A_count_resid_s{s}.npy" for s in (42, 7)])
f5 = avg([f"idea54_cache/A_future50_multi_s{s}.npy" for s in (42, 7)])
v66 = (.1824 * base + .2432 * hur + .0608 * mr + .1216 * od + .1520 * mo
       + .08 * cb + .08 * cr + .08 * f5)
B66 = sc(v66)
e1 = v66 - yv
B1 = float(np.mean(e1 ** 2))
log(f"v66local={B66:.2f} (base단독={sc(base):.2f})  B1={B1:.6f}")

# 학습은 라벨 유효행만. E[y|c]는 train 구간에서만 추정(누출 방지).
fit_mask = tr & (cls >= 0)
succ_by_cls = np.array([y[fit_mask & (cls == c)].mean() for c in range(5)])
log(f"E[y|c] (train추정) = {np.round(succ_by_cls, 5)}")
log(f"학습행 {fit_mask.sum():,} / train {tr.sum():,}")

w_rec = (0.5 ** ((UPTO - season) / 2.0))
fit_idx = np.where(fit_mask)[0]
n_es = int(len(fit_idx) * 0.92)
fi, ei = fit_idx[:n_es], fit_idx[n_es:]
va_i = np.where(va)[0]

CAT = dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5.0, verbose=100,
           random_seed=42, loss_function="MultiClass", classes_count=5,
           early_stopping_rounds=40, thread_count=max(1, (os.cpu_count() or 4) - 1))

log("5-class CatBoost 학습 시작...")
ts = time.time()
m = CatBoostClassifier(**CAT)
m.fit(X.iloc[fi], cls[fi], sample_weight=w_rec[fi], eval_set=(X.iloc[ei], cls[ei]))
log(f"학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)")

P = m.predict_proba(X.loc[va])          # (n_va, 5)
np.save(f"{CD}/A_proba5.npy", P)
p_mc = P @ succ_by_cls                   # P(success) = Σ P(c) E[y|c]
np.save(f"{CD}/A_mc.npy", p_mc)

Vp = np.mean((p_mc - r) ** 2)
k = np.mean((yv - r) * (p_mc - r)) / Vp
pc = r + k * (p_mc - r)
B2 = float(np.mean((pc - yv) ** 2))
rho = float(np.corrcoef(e1, pc - yv)[0, 1])
thr = np.sqrt(B1 / B2)
c66 = np.corrcoef(p_mc, v66)[0, 1]

print()
print("=" * 78)
print(f"5-class multiclass 단독 = {sc(p_mc):.2f}   (캘리브후 {sc(pc):.2f}, k={k:.3f})")
print(f"corr(v66) = {c66:.4f}   <- 기존 aux축은 전부 0.98대")
print(f"blend 손익분기: rho={rho:.6f}  임계={thr:.6f}  여유={thr-rho:+.6f}")
print()
bestw, bestd = 0.0, 0.0
for w in np.linspace(0, 0.5, 201):
    d = sc((1 - w) * v66 + w * pc) - B66
    if d > bestd:
        bestw, bestd = w, d
print(f"v66 blend 최적 w={bestw:.3f}  로컬델타={bestd:+.3f}")
for w in (0.05, 0.10, 0.15, 0.20, 0.30):
    print(f"   w={w:.2f} -> {sc((1-w)*v66 + w*pc)-B66:+8.3f}")
print()
print("클래스별 예측확률 평균 vs 실제비중:")
for c in range(5):
    print(f"  class{c}: 예측평균={P[:,c].mean():.4f} 실제={np.mean(cls[va]==c):.4f}")
log(f"총 {time.time()-t0:.0f}s")
