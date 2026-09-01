"""idea53 — '시즌후반 실적'을 aux head 타겟으로. (8-10월 데이터의 새 용도)

## 착안 경로
v60b(3-7월만 학습) 실측이 +0.05로 무변화 -> 8-10월 데이터가 **정보적으로 중복**임이 확인됨.
근거: 8-10월 행의 98.9%가 이미 3-7월에도 등장한 투수. 새 투수는 40명(5%)뿐.
      투수당 3-7월에 이미 1,221구 관측 -> 8-10월 832구 추가는 한계효용 낮음.
그런데 "예측용 표본으로서 중복"이라는 게 "타겟으로서 무가치"를 뜻하진 않는다.
**같은 데이터를 입력이 아니라 목표로 쓰면 완전히 다른 정보 경로가 된다.**

## 설계 (v49 formcast 계열, 실측 +3.41로 검증된 구조)
formcast: head1 = 향후 50투구 실현 성공률 (짧은 슬라이딩 윈도우, 표본 50)
본안    : head1 = **그 투수의 같은시즌 8-10월 전체 성공률** (표본 평균 832구)
          head2 = 같은시즌 8-10월 middle율 (제구붕괴 방향)
-> formcast보다 표본이 16배 커서 타겟 노이즈가 훨씬 작다.

## 왜 트리 분할이 좋아지는가
3-7월 행을 예측할 때 공유트리는 "이 투수가 시즌 후반에 어떻게 되는가"를 동시에
맞춰야 한다. 즉 '지금 잘 던지지만 후반에 무너지는 유형' vs '끝까지 유지하는 유형'을
가르는 split이 살아남는다. 이는 현재 피처(career/inseason/form)가 담지 못하는
**시즌 내 지속성(durability)** 축이다.

## Rule.md §4 준수
head는 train 행에만 부여(미래정보는 학습데이터 내부에서만 구성).
추론시 head0(y)만 사용 -> test 행은 자기 컬럼만 참조. v49와 동일 구조로 검증된 패턴.
8-10월 기록이 없는 투수(그 시즌 후반 미등판)는 NaN -> MultiRMSEWithMissingValues가 제외.

## 검증
fold A(train<=2023 -> 2024), 규약: (1-w)*v47local + w*신규, 전체2024 + 3-7월 병기.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea53_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
K_SMOOTH = 30.0  # 후반 표본이 적은 투수 축소용


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
mo = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                 usecols=["game_month"])["game_month"].to_numpy()
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def recover(col):
    c = np.round(meta[col].fillna(0).to_numpy(np.float64) * n_)
    co = c[order]
    d = np.empty(len(meta)); d[:-1] = co[1:] - co[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta)); lab[order] = d
    return lab


lab_rev = recover("asof_pitcher_reverse_rate")
lab_mid = recover("asof_pitcher_middle_rate")
valid = ~(np.isnan(lab_rev) | np.isnan(lab_mid))
tot = y + lab_rev + lab_mid
lab_other = np.where(valid, (tot == 0).astype(np.float64), np.nan)
h_mid = np.where(valid, 1.0 - lab_mid, np.nan)
h_other = np.where(valid, 1.0 - lab_other, np.nan)

log("시즌후반(8-10월) 실적 head 구성...")
g_glob = float(y.mean())
g_mid = float(np.nanmean(lab_mid))
late = mo >= 8
d = pd.DataFrame({"pid": pid, "season": seasons, "y": y, "mid": lab_mid, "late": late})
# 투수-시즌별 후반 집계 (train 내부 정보로만 구성)
agg = d[d["late"]].groupby(["pid", "season"]).agg(
    ls=("y", "sum"), ln=("y", "count"), lms=("mid", "sum"), lmn=("mid", "count"))
agg["late_succ"] = (agg["ls"] + K_SMOOTH * g_glob) / (agg["ln"] + K_SMOOTH)
agg["late_notmid"] = 1.0 - (agg["lms"] + K_SMOOTH * g_mid) / (agg["lmn"] + K_SMOOTH)
idx = pd.MultiIndex.from_arrays([pid, seasons])
h_late_succ = pd.Series(agg["late_succ"].reindex(idx).to_numpy())
h_late_notmid = pd.Series(agg["late_notmid"].reindex(idx).to_numpy())
# 후반 기록이 아예 없는 투수-시즌은 NaN 유지 (MultiRMSE가 제외)
h_late_succ = h_late_succ.to_numpy(np.float64)
h_late_notmid = h_late_notmid.to_numpy(np.float64)
cov = np.isfinite(h_late_succ)
log(f"  후반실적 커버리지 {cov.mean()*100:.1f}%  "
    f"late_succ mean={np.nanmean(h_late_succ):.4f} sd={np.nanstd(h_late_succ):.4f}")
log(f"  후반 미등판(NaN) {(~cov).sum():,}행 -> 해당 head만 자동 제외")

tr_m = seasons <= 2023
va_m = seasons == 2024
yv = y[va_m]; mv = mo[va_m]; seg37 = (mv >= 3) & (mv <= 7)
r = yv.mean(); BS = r * (1 - r)
w = 0.5 ** ((2023 - seasons) / 2.0)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)


def sc37(p):
    yy = yv[seg37]; rr = yy.mean()
    return 1e5 * (1 - np.mean((np.clip(p[seg37], 0, 1) - yy) ** 2) / (rr * (1 - rr)))


A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)
base = A([f"phase90_cache/A_base_{n}.npy" for n in ["d6", "d8", "sub"]])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
mr = A([f"idea13_cache/A_multires_s{k}.npy" for k in [42, 7]])
od = A([f"idea13_cache/A_ordinal_s{k}.npy" for k in [42, 7]])
V47 = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
B_full, B_37 = sc(V47), sc37(V47)
uni_ref = A([f"idea46_cache/A_midother_s{k}.npy" for k in [42, 7]])
log(f"v47local 전체={B_full:.2f} 3-7월={B_37:.2f}  (참고 midother Δ: "
    f"{sc(0.9*V47+0.1*uni_ref)-B_full:+.2f} / {sc37(0.9*V47+0.1*uni_ref)-B_37:+.2f})")

CAT = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)
n_es = int(tr_m.sum() * 0.92)

VARIANTS = {
    # 순수 후반축만 (formcast 계열 단독 효과 측정)
    "late2": [y, h_late_succ, h_late_notmid],
    # 검증된 midother에 후반축을 얹은 것 (실전 후보)
    "mid_other_late": [y, h_mid, h_other, h_late_succ, h_late_notmid],
    # 후반 성공률만 (최소 구성)
    "late_succ_only": [y, h_late_succ],
}

print()
print("=" * 84)
print(f"{'변종':<18}{'head수':>7}{'단독':>9}{'시드폭':>8}{'전체Δ':>9}{'3-7월Δ':>9}{'예상실측':>10}")
print("=" * 84)
for name, heads in VARIANTS.items():
    Ymat = np.column_stack([np.asarray(h, dtype=np.float64) for h in heads])
    ps = []
    for seed in SEEDS:
        f = f"{CD}/A_{name}_s{seed}.npy"
        if os.path.exists(f):
            ps.append(np.load(f)); continue
        ts = time.time()
        m = CatBoostRegressor(**CAT, random_seed=seed)
        m.fit(X.loc[tr_m].iloc[:n_es], Ymat[tr_m][:n_es], sample_weight=w[tr_m][:n_es],
              eval_set=(X.loc[tr_m].iloc[n_es:], Ymat[tr_m][n_es:]))
        p = np.clip(m.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
        np.save(f, p); ps.append(p)
        log(f"    [{name}/s{seed}] best_iter={m.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
    u = np.mean(ps, axis=0)
    spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
    d_full = sc(0.90 * V47 + 0.10 * u) - B_full
    d_37 = sc37(0.90 * V47 + 0.10 * u) - B_37
    pred = 10.62 + 3.82 * d_37   # 3-7월 캘리브 (Pearson 0.921)
    print(f"{name:<18}{Ymat.shape[1]:7d}{sc(u):9.2f}{spread:8.2f}{d_full:+9.2f}{d_37:+9.2f}{pred:+10.2f}")

print()
print("기준: midother(2head) 3-7월Δ=-1.43 -> 실측 +3.25 / 캘리브 실측Δ=10.62+3.82xΔ37")
print("판정: 3-7월Δ가 midother(-1.43)보다 높고 시드폭 이내면 프로덕션 후보")
log(f"총 {time.time()-t0:.0f}s")
