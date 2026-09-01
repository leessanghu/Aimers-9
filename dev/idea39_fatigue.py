"""아이디어39 — 야구 도메인: 등판 내 피로(within-outing fatigue) + 등판 간 회복.

현재 162피처의 결정적 공백:
  `inning`, `role_med_inning`, `role_late_share`는 있지만
  **"이 투수가 오늘 이 등판에서 지금까지 몇 구를 던졌는가"가 없다.**
  세이버메트릭스에서 가장 확립된 투수 성능 저하 변수 두 가지가 통째로 빠져 있음:
    (1) 투구수 절벽 — 선발은 75구 부근부터 저하, 100구 넘으면 급락
    (2) TTO(Times Through the Order) 페널티 — 같은 타자를 3번째 상대할 때
        급격히 불리해짐. 야구 분석에서 가장 재현성 높은 발견 중 하나.
  둘 다 '경기 내 상태'라서 시즌 누적 통계(asof_*)로는 절대 대체 안 됨.

등판 경계: formfeat.build_role_table과 동일 근사
  (season, game_month, game_dayofweek) 변경 또는 inning 감소 -> 새 등판.
  train.csv에 game_id가 없어서 쓰는 근사.

leakage: 모든 값이 '이 행 이전의 같은 등판/과거 등판'만 사용(expanding). 미래 참조 없음.
스크리닝: partial_gain + **Spearman 단조중복 게이트**(idea38에서 위양성 잡은 뒤 추가).
분류: '새 정보 추가'류 (실측 4/4 성공 계열).
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

t0 = time.time()
N_REF = 253507
SIGMA = 1e5 * (1.0 / N_REF)


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
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
raw = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                  usecols=["season", "game_month", "game_dayofweek", "inning",
                           "pitcher_id", "batter_id"])
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

d = raw.copy()
d["row_num"] = meta["row_num"].to_numpy()
d = d.sort_values(["pitcher_id", "row_num"])
pid = d["pitcher_id"].to_numpy()
key = (d["season"].to_numpy() * 10000 + d["game_month"].to_numpy() * 100
       + d["game_dayofweek"].to_numpy())
inn = d["inning"].to_numpy()
new_p = np.empty(len(d), bool); new_p[0] = True; new_p[1:] = pid[1:] != pid[:-1]
new_k = np.empty(len(d), bool); new_k[0] = True; new_k[1:] = key[1:] != key[:-1]
drop = np.empty(len(d), bool); drop[0] = False; drop[1:] = inn[1:] < inn[:-1]
d["_app"] = np.cumsum(new_p | new_k | drop)
log(f"  추정 등판 수 {d['_app'].nunique():,}  (투수-시즌당 평균 "
    f"{d['_app'].nunique()/d.groupby(['pitcher_id','season']).ngroups:.1f})")

g = d.groupby("_app", sort=False)

# ---- 등판 내 피로 ----
d["fat_pitch_in_outing"] = g.cumcount().astype(np.float64)          # 오늘 누적 투구수
d["fat_inn_in_outing"] = d["inning"].to_numpy() - g["inning"].transform("min").to_numpy()
# TTO: 이번 등판에서 이 타자를 몇 번째로 상대하는가 (0=처음)
d["fat_tto"] = d.groupby(["_app", "batter_id"], sort=False).cumcount().astype(np.float64)
# 이번 등판 상대한 고유타자 수(진행 중 누적) — 타순 한 바퀴 근사
d["fat_bf"] = (d.groupby(["_app", "batter_id"], sort=False).cumcount() == 0).astype(np.float64)
d["fat_bf"] = g["fat_bf"].cumsum() - 1.0
# 도메인 임계 플래그
d["fat_over75"] = (d["fat_pitch_in_outing"] >= 75).astype(np.float64)
d["fat_over100"] = (d["fat_pitch_in_outing"] >= 100).astype(np.float64)
d["fat_tto3"] = (d["fat_tto"] >= 2).astype(np.float64)              # 3번째 이상 대면

# ---- 등판 간 회복/누적 ----
app_tot = g.size().rename("app_pitches")
app_first = g["row_num"].min().rename("app_first_row")
app_tbl = pd.concat([app_tot, app_first], axis=1).reset_index()
app_tbl["pitcher_id"] = g["pitcher_id"].first().to_numpy()
app_tbl["season"] = g["season"].first().to_numpy()
app_tbl = app_tbl.sort_values(["pitcher_id", "app_first_row"])
ag = app_tbl.groupby("pitcher_id", sort=False)
app_tbl["prev_pitches"] = ag["app_pitches"].shift(1)
app_tbl["prev_gap"] = app_tbl["app_first_row"] - ag["app_first_row"].shift(1)
app_tbl["roll3"] = ag["app_pitches"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).sum())
mp = app_tbl.set_index("_app")
d["fat_prev_pitches"] = mp["prev_pitches"].reindex(d["_app"]).to_numpy()
d["fat_prev_gap"] = np.log1p(mp["prev_gap"].reindex(d["_app"]).to_numpy())
d["fat_roll3_pitches"] = mp["roll3"].reindex(d["_app"]).to_numpy()
# 짧은 휴식 x 많은 직전투구 (회복 부족)
d["fat_short_rest_load"] = d["fat_prev_pitches"] / np.maximum(np.expm1(d["fat_prev_gap"]), 1.0)

COLS = ["fat_pitch_in_outing", "fat_inn_in_outing", "fat_tto", "fat_bf",
        "fat_over75", "fat_over100", "fat_tto3",
        "fat_prev_pitches", "fat_prev_gap", "fat_roll3_pitches", "fat_short_rest_load"]
feats = d[COLS].sort_index()
log(f"  피처 {len(COLS)}개 구성 완료")
log(f"  등판당 투구수 분포 p50={d.groupby('_app').size().median():.0f} "
    f"p90={d.groupby('_app').size().quantile(0.9):.0f} max={d.groupby('_app').size().max()}")

va = seasons == 2024
b = np.mean([np.load(f"phase90_cache/A_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
h = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) * np.load(f"phase90_cache/A_snc_{n}.npy")
             for n in ["d6", "d8"]], axis=0)
m = np.mean([np.load(f"idea13_cache/A_multires_s{k}.npy") for k in [42, 7]], axis=0)
o = np.mean([np.load(f"idea13_cache/A_ordinal_s{k}.npy") for k in [42, 7]], axis=0)
dd = np.mean([np.load(f"idea31_cache/A_midaxis_s{k}.npy") for k in [42, 7]], axis=0)
p_base = np.clip(0.20 * b + 0.40 * h + 0.10 * m + 0.20 * o + 0.10 * dd, 1e-6, 1 - 1e-6)
yv = y[va]
mv = raw["game_month"].to_numpy()[va]
seg37 = (mv >= 3) & (mv <= 7)

log("Spearman 단조중복 게이트 계산...")
Xv = {c: X[c].to_numpy()[va] for c in X.columns}
print()
print("=" * 96)
print(f"{'후보':<24}{'전체2024':>11}{'3-7월':>10}{'부호':>6}{'최대|Spearman|':>15}{'중복상대':<22}{'판정':>8}")
print(f"(1시그마={SIGMA:.2f}, 통과=4시그마={4*SIGMA:.1f} / Spearman>0.99면 트리에 중복->즉시기각)")
print("-" * 96)
res = []
for c in COLS:
    z = feats[c].to_numpy()[va]
    if not np.isfinite(z).any() or np.nanstd(z) == 0:
        continue
    zz = np.nan_to_num(z, nan=float(np.nanmedian(z)))
    g_all, _ = partial_gain(yv, p_base, zz)
    g_37, pc37 = partial_gain(yv[seg37], p_base[seg37], zz[seg37])
    bestr, bestc = 0.0, "-"
    for k, v in Xv.items():
        try:
            r = abs(spearmanr(zz, v).statistic)
        except Exception:
            continue
        if np.isfinite(r) and r > bestr:
            bestr, bestc = r, k
    dup = bestr > 0.99
    ok = (g_37 > 4 * SIGMA) and not dup
    verdict = "기각(중복)" if dup else ("통과" if ok else "기각")
    res.append((c, g_all, g_37, bestr, verdict))
    print(f"{c:<24}{g_all:11.2f}{g_37:10.2f}{('+' if pc37>0 else '-'):>6}{bestr:15.4f}  {bestc:<20}{verdict:>8}")
print()
pas = [r for r in res if r[4] == "통과"]
print(f"통과 {len(pas)}개: {[r[0] for r in pas]}")
print("참고: 채택된 bat_inseason_smooth=17.05(6.6시그마) / midaxis축 실측+7.72")
log(f"총 {time.time()-t0:.0f}s")
