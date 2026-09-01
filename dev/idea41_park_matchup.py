"""idea41 — 현재 162피처에 구조적으로 없는 도메인 축 5종.

발굴 근거: top_bottom='T'(초)면 투수가 홈팀(데이터에서 100% 확인).
따라서 **구장 = 홈팀 구장**을 도출할 수 있는데 지금 피처엔 구장 개념이 아예 없다.
pitcher_team_id_te / batter_team_id_te는 있지만, 트리가
"top_bottom이 T면 pitcher_team을, B면 batter_team을 구장으로 보라"는
깊은 상호작용을 스스로 만들어야 해서 사실상 못 쓴다. 명시적으로 준다.

축 1) 구장 효과(park) — 마운드 상태/배경/조명은 제구에 직접 영향(도메인 통설).
축 2) 구장 친숙도 — 이 투수가 이 구장에서 던진 누적 경험(홈 마운드 적응).
축 3) 팀레벨 매치업 — 개인 h2h는 phase50에서 실패했으나(대전 40%가 n=0)
      투수 vs 상대'팀'은 표본이 훨씬 크다. 통계적으로 훨씬 건전한 버전.
축 4) 홈/원정 x 실력 — 홈어드밴티지의 투수별 이질성.
축 5) Marcel식 3년 가중투영 — 현재는 career(등가중)와 ly(1년)뿐.
      실제 야구 투영의 표준은 최근 3년을 5/4/3 가중 + 평균회귀.

leakage 안전: 모든 테이블은 train에서만 만들고 **season-1 조회**만 한다
(platoon.py/lastyear.py와 동일 패턴). test 행 간 참조 없음 -> Rule §4 준수.
스크리닝: partial_gain + Spearman 단조중복 게이트. 분류='새 정보 추가'류.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def partial_gain(y, p, z):
    z = np.nan_to_num(np.asarray(z, np.float64), nan=0.0)
    if z.std() == 0:
        return 0.0, 0.0
    A = np.column_stack([np.ones(len(y)), p])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rz = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


log("로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
raw = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                  usecols=["season", "game_month", "top_bottom", "pitcher_id", "batter_id",
                           "pitcher_team_id", "batter_team_id", "control_success"])
y = raw["control_success"].to_numpy(np.float64)
seasons = raw["season"].to_numpy(np.float64)
mo = raw["game_month"].to_numpy()
g_rate = float(y.mean())
sr = sorted(np.unique(seasons).tolist())

is_home_p = (raw["top_bottom"].to_numpy() == "T")
park = np.where(is_home_p, raw["pitcher_team_id"].to_numpy(), raw["batter_team_id"].to_numpy())
raw["park"] = park
raw["is_home_p"] = is_home_p.astype(np.float64)
log(f"  구장 {pd.Series(park).nunique()}개, 홈투수 비율 {is_home_p.mean()*100:.1f}%")

K = 200.0  # 구장 셀은 크므로 완만한 축소


def s1_lookup(tbl, keys, prior):
    """season-1 누적 테이블에서 조회 (없으면 prior)."""
    idx = pd.MultiIndex.from_arrays([*keys[:-1], keys[-1] - 1])
    v = pd.Series(tbl.reindex(idx).to_numpy())
    return v.fillna(prior).to_numpy(np.float64) if np.isscalar(prior) else v.fillna(pd.Series(prior)).to_numpy(np.float64)


F = pd.DataFrame(index=X.index)

# ---- 축1: 구장 효과 (season-1까지 누적 성공률, K축소) ----
gp = raw.groupby(["park", "season"])["control_success"].agg(s="sum", n="count").sort_index()
cum = gp.groupby(level=0).cumsum()
rate = ((cum["s"] + K * g_rate) / (cum["n"] + K))
piv = rate.unstack().reindex(columns=sr).ffill(axis=1).stack(future_stack=True)
F["park_rate"] = s1_lookup(piv, [raw["park"], raw["season"]], g_rate)
F["park_dev"] = F["park_rate"] - g_rate
npiv = cum["n"].unstack().reindex(columns=sr).ffill(axis=1).stack(future_stack=True)
F["park_n"] = np.log1p(s1_lookup(npiv, [raw["park"], raw["season"]], 0.0))

# ---- 축2: 이 투수의 이 구장 경험 (season-1까지 누적 투구수) ----
pp = raw.groupby(["pitcher_id", "park", "season"]).size().rename("n").sort_index()
ppc = pp.groupby(level=[0, 1]).cumsum()
ppiv = ppc.unstack().reindex(columns=sr).ffill(axis=1).stack(future_stack=True)
F["park_fam"] = np.log1p(s1_lookup(ppiv, [raw["pitcher_id"], raw["park"], raw["season"]], 0.0))

# ---- 축3: 투수 vs 상대팀 (팀레벨 h2h, season-1 누적) ----
vt = raw.groupby(["pitcher_id", "batter_team_id", "season"])["control_success"].agg(s="sum", n="count").sort_index()
vtc = vt.groupby(level=[0, 1]).cumsum()
KV = 100.0
# 투수 자신의 커리어율로 축소 (전역이 아니라 개인 기준선)
own = raw.groupby(["pitcher_id", "season"])["control_success"].agg(s="sum", n="count").sort_index()
ownc = own.groupby(level=0).cumsum()
own_rate = ((ownc["s"] + 50 * g_rate) / (ownc["n"] + 50)).unstack().reindex(columns=sr).ffill(axis=1).stack(future_stack=True)
p_prior = s1_lookup(own_rate, [raw["pitcher_id"], raw["season"]], g_rate)
vt_s = s1_lookup(vtc["s"].unstack().reindex(columns=sr).ffill(axis=1).stack(future_stack=True),
                 [raw["pitcher_id"], raw["batter_team_id"], raw["season"]], 0.0)
vt_n = s1_lookup(vtc["n"].unstack().reindex(columns=sr).ffill(axis=1).stack(future_stack=True),
                 [raw["pitcher_id"], raw["batter_team_id"], raw["season"]], 0.0)
F["vsteam_dev"] = (vt_s + KV * p_prior) / (vt_n + KV) - p_prior
F["vsteam_n"] = np.log1p(vt_n)

# ---- 축4: 홈/원정 x 실력 ----
F["is_home_p"] = raw["is_home_p"].to_numpy()
abil = X["inseason_success_smooth"].to_numpy()
F["home_x_ability"] = F["is_home_p"].to_numpy() * abil
# 투수별 홈/원정 스플릿 (season-1)
hs = raw.groupby(["pitcher_id", "is_home_p", "season"])["control_success"].agg(s="sum", n="count").sort_index()
hsc = hs.groupby(level=[0, 1]).cumsum()
h_s = s1_lookup(hsc["s"].unstack().reindex(columns=sr).ffill(axis=1).stack(future_stack=True),
                [raw["pitcher_id"], raw["is_home_p"], raw["season"]], 0.0)
h_n = s1_lookup(hsc["n"].unstack().reindex(columns=sr).ffill(axis=1).stack(future_stack=True),
                [raw["pitcher_id"], raw["is_home_p"], raw["season"]], 0.0)
F["homesplit_dev"] = (h_s + 100 * p_prior) / (h_n + 100) - p_prior

# ---- 축5: Marcel식 3년 가중투영 (5/4/3 가중 + 평균회귀) ----
ys = raw.groupby(["pitcher_id", "season"])["control_success"].agg(s="sum", n="count")
sp = ys["s"].unstack().reindex(columns=sr)
npv = ys["n"].unstack().reindex(columns=sr)
mS = np.zeros_like(sp.to_numpy(), dtype=np.float64)
mN = np.zeros_like(npv.to_numpy(), dtype=np.float64)
S_, N_ = np.nan_to_num(sp.to_numpy()), np.nan_to_num(npv.to_numpy())
for j in range(len(sr)):
    for lag, wt in [(1, 5.0), (2, 4.0), (3, 3.0)]:
        if j - lag >= 0:
            mS[:, j] += wt * S_[:, j - lag]
            mN[:, j] += wt * N_[:, j - lag]
KM = 600.0
marcel = (mS + KM * g_rate) / (mN + KM)
mtbl = pd.DataFrame(marcel, index=sp.index, columns=sr).stack(future_stack=True)
idxm = pd.MultiIndex.from_arrays([raw["pitcher_id"], raw["season"]])
F["marcel3"] = pd.Series(mtbl.reindex(idxm).to_numpy()).fillna(g_rate).to_numpy(np.float64)
F["marcel_minus_career"] = F["marcel3"] - X["asof_pitcher_success_rate_smooth"].to_numpy()

F = F.astype(np.float64).replace([np.inf, -np.inf], np.nan).fillna(0.0)
log(f"  후보 {len(F.columns)}개 구성")

# ---- 스크리닝 (fold A 2024) ----
va = seasons == 2024
b = np.mean([np.load(f"phase90_cache/A_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
h = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) * np.load(f"phase90_cache/A_snc_{n}.npy")
             for n in ["d6", "d8"]], axis=0)
mm = np.mean([np.load(f"idea13_cache/A_multires_s{k}.npy") for k in [42, 7]], axis=0)
oo = np.mean([np.load(f"idea13_cache/A_ordinal_s{k}.npy") for k in [42, 7]], axis=0)
dm = np.mean([np.load(f"idea31_cache/A_midaxis_s{k}.npy") for k in [42, 7]], axis=0)
p_base = np.clip(0.20 * b + 0.40 * h + 0.10 * mm + 0.20 * oo + 0.10 * dm, 1e-6, 1 - 1e-6)
yv = y[va]
mv = mo[va]
seg = (mv >= 3) & (mv <= 7)
n_all, n37 = int(va.sum()), int(seg.sum())

Xv = {c: X[c].to_numpy()[va] for c in X.columns}
print()
print("=" * 100)
print(f"{'후보':<22}{'전체g':>9}{'σ':>7}{'3-7월g':>9}{'σ':>7}{'부호':>6}{'최대Spear':>11}{'중복상대':<20}{'판정':>8}")
print(f"(1단계 관대기준 3.2σ / Spearman>0.99 즉시기각)   n_all={n_all:,} n37={n37:,}")
print("-" * 100)
surv = []
for c in F.columns:
    z = F[c].to_numpy()[va]
    ga, _ = partial_gain(yv, p_base, z)
    g3, pc3 = partial_gain(yv[seg], p_base[seg], z[seg])
    ka, k3 = np.sqrt(ga * n_all / 1e5), np.sqrt(g3 * n37 / 1e5)
    br, bc = 0.0, "-"
    for k, v in Xv.items():
        try:
            r = abs(spearmanr(z, v).statistic)
        except Exception:
            continue
        if np.isfinite(r) and r > br:
            br, bc = r, k
    dup = br > 0.99
    ok = (k3 >= 3.2) and not dup
    if ok:
        surv.append(c)
    print(f"{c:<22}{ga:9.2f}{ka:7.2f}{g3:9.2f}{k3:7.2f}{('+' if pc3>0 else '-'):>6}{br:11.3f}  {bc:<18}"
          f"{('기각(중복)' if dup else ('통과' if ok else '기각')):>8}")
print()
print(f"통과: {surv if surv else '없음'}")
print("벤치마크: bat_inseason_smooth=6.6σ(채택) / streak_hot_flag=4.07σ(검증중)")
log(f"총 {time.time()-t0:.0f}s")
