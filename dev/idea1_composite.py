"""아이디어 1 — 주축 합성피처 완성 (x_ability_here 에 빠진 축 채우기).

관찰 (phase94 심층분석):
    SHAP 1위 x_ability_here (0.05129, 2위의 1.6배) = inseason + platoon_diff + inning_diff
    확률 스케일에서 맥락보정을 미리 더한 합성피처. split 91회, threshold가 0.41~0.59 전구간.

문제:
    합성에 들어간 것: platoon(std 0.0118), inning(std 0.0080)  <- 제일 작은 보정 두 개
    빠진 것:          bat_inseason_smooth(std 0.0463, SHAP 3위!), count_diff(std 0.0058)
    타자 축이 platoon보다 4배 큰 신호인데 주축 합성에서 빠져 있다.

가설:
    이건 '정보 추가'가 아니라 '용량 절약' 아이디어다. 합성피처는 기존 컬럼의 합이라
    정보량 증분은 0이어야 한다(cmd_index와 동일). 그러나 x_ability_here 자신이 순수
    합성인데 SHAP 1위라는 사실이, 트리가 이 합을 split으로 재구성하는 비용이 크다는 증거다.

    -> 정보 스크리너는 '예상대로 0인지' 확인용. 진짜 판정은 3폴드 재학습.

실험:
    Part1  분할반 정보 스크리닝 (참조피처 대비). ~0 나와야 가설과 정합.
    Part2  3폴드(A/C/B) 재학습: base(HGB3변종) vs base+신규합성. 동일 마스크/가중치.
    Part3  fold A 모델에서 신규피처 SHAP magnitude -> 실제로 쓰이는지.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea1_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def _clean(z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    return z


def residualize_np(y, p, n_bins=200):
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1])
    codes = np.searchsorted(qs, p).astype(np.int64)
    sums = np.bincount(codes, weights=y, minlength=n_bins)
    cnts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    means = np.divide(sums, cnts, out=np.zeros(n_bins), where=cnts > 0)
    return y - means[codes]


def splithalf_gain(ry, z, n_bins=12, n_splits=5):
    z = _clean(z)
    if z.std() == 0:
        return 0.0, 0.0
    qs = np.unique(np.quantile(z, np.linspace(0, 1, n_bins + 1)[1:-1]))
    if len(qs) == 0:
        return 0.0, 0.0
    codes = np.searchsorted(qs, z).astype(np.int64)
    nb = codes.max() + 1
    vary = ry.var()
    gains = []
    for s in range(n_splits):
        rs = np.random.RandomState(1000 + s)
        half = rs.rand(len(ry)) < 0.5
        sums = np.bincount(codes[half], weights=ry[half], minlength=nb)
        cnts = np.bincount(codes[half], minlength=nb).astype(np.float64)
        means = np.where(cnts > 0, np.divide(sums, np.maximum(cnts, 1)), ry[half].mean())
        yb = ry[~half]
        gains.append(1e5 * (1.0 - np.mean((yb - means[codes[~half]]) ** 2) / yb.var()) * (yb.var() / vary))
    return float(np.mean(gains)), float(np.std(gains, ddof=1))


def logit(p, eps=1e-4):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
g = float(meta["global_rate"].iloc[0])
log(f"  X={X.shape}  전역성공률={g:.4f}")

# phase90과 동일한 step 마스크 (라벨복원 가능 행)
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr]
step = np.zeros(len(meta), dtype=bool)
step[ordr[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr]) == 1)
log(f"  step 마스크 {step.sum():,}행 ({100*step.mean():.2f}%)")

# ---------------- 신규 합성피처 ----------------
here = X["x_ability_here"].to_numpy(np.float64)
bat = X["bat_inseason_smooth"].to_numpy(np.float64)
cntd = X["count_diff"].to_numpy(np.float64)
ptd = X["pt_dev"].to_numpy(np.float64)
bplat = X["bplatoon_diff"].to_numpy(np.float64)
abil = X["inseason_success_smooth"].to_numpy(np.float64)
plat = X["platoon_diff"].to_numpy(np.float64)
innd = X["inning_diff"].to_numpy(np.float64)

NEW = {}
NEW["xh_bat"] = here + (bat - g)                                   # 타자축 추가
NEW["xh_count"] = here + cntd                                      # 카운트축 추가
NEW["xh_full"] = here + cntd + (bat - g)                           # 둘 다
NEW["xh_full2"] = here + cntd + (bat - g) + ptd + bplat            # 전부
NEW["xh_margin"] = here - bat                                      # 투타 대결 마진
NEW["xh_logit"] = logit(abil) + (logit(abil + plat) - logit(abil)) \
                  + (logit(abil + innd) - logit(abil)) \
                  + (logit(abil + cntd) - logit(abil)) \
                  + (logit(bat) - logit(g))                        # 로짓공간 합성
log(f"신규 합성피처 {len(NEW)}개 생성")
for k, v in NEW.items():
    print(f"    {k:<12} mean={v.mean():.4f} std={v.std():.4f} corr(x_ability_here)={np.corrcoef(v,here)[0,1]:.4f}")

# ---------------- Part 1: 정보 스크리닝 ----------------
log("[Part1] 분할반 정보 스크리닝 (fold A 잔차 기준)...")
zc = np.load("phase67_cache/phase69_preds.npz")
p_A = 0.5 * zc["hgb"] + 0.5 * zc["cat3"]
y_A = zc["y"].astype(np.float64)
vaA = seasons == 2024
ry = residualize_np(y_A, p_A, 200)

REF = {
    "ref_x_ability_here": here[vaA],
    "ref_bat_inseason": bat[vaA],
    "ref_inseason_success": abil[vaA],
    "ref_count_diff": cntd[vaA],
    "ref_platoon_diff": plat[vaA],
}
rows = []
for nm, v in REF.items():
    sh, sd = splithalf_gain(ry, v)
    rows.append(dict(group="참조", feature=nm, gain=sh, sd=sd))
for nm, v in NEW.items():
    sh, sd = splithalf_gain(ry, v[vaA])
    rows.append(dict(group="신규", feature=nm, gain=sh, sd=sd))
scr = pd.DataFrame(rows)
print()
print(f"{'':<22}{'분할반gain':>12}{'SD':>8}")
for _, r in scr.iterrows():
    tag = "[참조]" if r.group == "참조" else "[신규]"
    print(f"{tag}{r.feature:<16}{r.gain:12.2f}{r.sd:8.2f}")
ref_mean = scr[scr.group == "참조"].gain.mean()
new_max = scr[scr.group == "신규"].gain.max()
print(f"\n참조 평균 {ref_mean:+.2f} / 신규 최대 {new_max:+.2f}")
print("=> 예상대로 정보증분 ~0 (합성피처이므로 당연). 진짜 판정은 Part2 재학습." if new_max < 15
      else "=> 예상 밖으로 정보증분 있음. 해석 주의.")

# ---------------- Part 2: 3폴드 재학습 ----------------
HGB_VARIANTS = [
    ("d6", dict(max_depth=6, max_leaf_nodes=31, random_state=42)),
    ("d8", dict(max_depth=8, max_leaf_nodes=15, random_state=2024)),
    ("sub", dict(max_depth=6, max_leaf_nodes=31, max_features=0.6, random_state=123)),
]
BASE_HGB = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20)

# 어떤 조합을 모델에 넣을지: 정보 중복 최소화 위해 대표 3개
ADD_SETS = {
    "A_batonly": ["xh_bat"],
    "B_full": ["xh_full"],
    "C_full_margin": ["xh_full", "xh_margin"],
    "D_logit": ["xh_logit"],
}


def run_fold(train_upto, valid_season, tag):
    log(f"===== fold {tag}: train<={train_upto} -> valid={valid_season} =====")
    tr_m = (seasons <= train_upto) & step
    va_m = seasons == valid_season
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((train_upto - seasons) / 2.0)

    def score(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    out = {}
    # 기준선 (phase90 캐시 재사용)
    base = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n, _ in HGB_VARIANTS], axis=0)
    out["base"] = score(base)
    log(f"  base(캐시) = {out['base']:.2f}")

    for setname, cols in ADD_SETS.items():
        preds = []
        for vn, extra in HGB_VARIANTS:
            f = f"{CD}/{tag}_{setname}_{vn}.npy"
            if os.path.exists(f):
                preds.append(np.load(f))
                continue
            Xa = X.copy()
            for c in cols:
                Xa[c] = NEW[c]
            p = dict(BASE_HGB); p.update(extra)
            ts = time.time()
            m = HistGradientBoostingClassifier(**p).fit(Xa.loc[tr_m], y[tr_m], sample_weight=w[tr_m])
            pr = m.predict_proba(Xa.loc[va_m])[:, 1]
            np.save(f, pr)
            preds.append(pr)
            log(f"    {setname}/{vn} 완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)")
            del Xa
        out[setname] = score(np.mean(preds, axis=0))
        log(f"  {setname} = {out[setname]:.2f}  (base대비 {out[setname]-out['base']:+.2f})")
    return out


results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    results[tag] = run_fold(upto, val, tag)

print()
print("=" * 76)
print(f"{'조합':<16}" + "".join(f"{t:>18}" for t in ["A(2024)", "C(2022)", "B(2023)"]) + f"{'최소이득':>10}")
print("-" * 76)
for setname in ["base"] + list(ADD_SETS):
    row = f"{setname:<16}"
    gains = []
    for t in ["A", "C", "B"]:
        v = results[t][setname]
        d = v - results[t]["base"]
        gains.append(d)
        row += f"{v:12.2f}({d:+5.2f})"
    row += f"{min(gains):10.2f}" if setname != "base" else f"{0.0:10.2f}"
    print(row)

best = max(ADD_SETS, key=lambda s: min(results[t][s] - results[t]["base"] for t in ["A", "C", "B"]))
best_min = min(results[t][best] - results[t]["base"] for t in ["A", "C", "B"])
print()
print(f"3폴드 최소이득 최대인 조합: {best}  (최소 {best_min:+.2f})")
print("=> 채택 검토 (3폴드 모두 개선)" if best_min > 2 else "=> 기각 (3폴드 일관성 없음)")
pd.DataFrame(results).to_csv("idea1_results.csv")
log(f"총 {time.time()-t0:.0f}s")
