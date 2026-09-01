"""phase82 — Trackman 미사용 컬럼에서 '실행 실패' 신호 발굴.

현행 trackman_profile.py의 USECOLS에는 tagged_pitch_type / auto_pitch_type / game_date /
pitch_of_pa 가 아예 없다(로드조차 안 함). 17개 피처는 전부 물리량의 평균/SD다.

핵심 가설 (tagged vs auto 불일치):
    tagged_pitch_type = 사람이 라벨한 '의도한 구종'
    auto_pitch_type   = 물리량으로 자동 분류된 '실제로 나온 구종'
    둘이 불일치 = 던지려던 구질이 안 나왔다 = 실행 실패
    control_success('의도한 곳에 갔는가')와 개념적으로 같은 축인데, 물리량 평균과 달리
    '의도 대비 결과'를 직접 관측한 값이다. Domain.md가 말한 'stuff가 아니라 command'.

부가 후보:
    zone_speed 완전 미사용 -> rel_speed - zone_speed = 비행 중 감속(항력/회전효율)
    auto_pitch_type 기반 레퍼토리 엔트로피 (arsenal_entropy.py는 공식 asof mix 3종만 씀)
    pitch_of_pa 미사용

규칙: 전부 (투수, season-1) 누적 룩업. trackman은 2019~2024만 존재해 2025 유입 불가.
방법: phase64b partial_gain (자유도 1, 귀무편향 0.4점). 4점=1시그마, 16점=2시그마.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

TM_PATH = "../data/trackman_history.csv"
MAP_PATH = "pitcher_map.csv"
VALID_SEASON = 2024
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def partial_gain(y, p, z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        if (~bad).sum() == 0:
            return 0.0, 0.0
        z[bad] = np.nanmedian(z[~bad])
    if z.std() == 0:
        return 0.0, 0.0
    A = np.column_stack([np.ones(len(y)), p])
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rz = z - A @ np.linalg.lstsq(A, z, rcond=None)[0]
    if rz.std() == 0:
        return 0.0, 0.0
    pc = float(np.corrcoef(ry, rz)[0, 1])
    return 1e5 * pc ** 2, pc


def block_gain(y, p, Z):
    Z = np.column_stack([np.nan_to_num(Z[:, j], nan=np.nanmedian(Z[:, j])) for j in range(Z.shape[1])])
    Z = Z[:, [j for j in range(Z.shape[1]) if Z[:, j].std() > 0]]
    n, k = len(y), Z.shape[1]
    X0 = np.column_stack([np.ones(n), p])
    X1 = np.column_stack([X0, Z])

    def r2(X):
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - (y - X @ c).var() / y.var()

    return 1e5 * (r2(X1) - r2(X0) - k / n), k


log("train + 예측 캐시 로드...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                 usecols=["season", "pitcher_id", "control_success"])
z = np.load("phase67_cache/phase69_preds.npz")
p_gbdt = 0.5 * z["hgb"] + 0.5 * z["cat3"]
y_va = z["y"].astype(np.float64)
va = (df["season"] == VALID_SEASON).to_numpy()
assert va.sum() == len(y_va)
log(f"valid={va.sum():,}  기준 잠재력={1e5*np.corrcoef(p_gbdt,y_va)[0,1]**2:.1f}")

log("trackman 로드 (신규 컬럼 포함)...")
USE = ["season", "trackman_game_id", "pitch_no", "pitch_of_pa", "balls_before", "strikes_before",
       "pitcher_trackman_id", "tagged_pitch_type", "auto_pitch_type", "pitch_type_group",
       "rel_speed", "zone_speed", "spin_rate", "induced_vert_break", "horz_break"]
m = pd.read_csv(MAP_PATH).sort_values("sim", ascending=False).drop_duplicates("tm_id")
t2p = m.set_index("tm_id")["pitcher_id"]
tm = pd.read_csv(TM_PATH, encoding="utf-8-sig", usecols=USE)
tm = tm.rename(columns={"pitcher_trackman_id": "tm_id"})
tm["pitcher_id"] = tm["tm_id"].map(t2p)
tm = tm.dropna(subset=["pitcher_id"])
tm["pitcher_id"] = tm["pitcher_id"].astype(np.int64)
log(f"  trackman {len(tm):,}행  투수 {tm.pitcher_id.nunique()}명")

# ---- 신규 파생 ----
tg = tm["tagged_pitch_type"].astype(str).str.strip().str.lower()
au = tm["auto_pitch_type"].astype(str).str.strip().str.lower()
valid = (~tg.isin(["nan", "undefined", "other", ""])) & (~au.isin(["nan", "undefined", "other", ""]))
tm["disagree"] = np.where(valid, (tg != au).astype(float), np.nan)
tm["velo_loss"] = tm["rel_speed"] - tm["zone_speed"]
tm["velo_retain"] = tm["zone_speed"] / tm["rel_speed"].replace(0, np.nan)
tm["is_3ball"] = (tm["balls_before"] >= 3).astype(float)
tm["deep_pa"] = (tm["pitch_of_pa"] >= 4).astype(float)
log(f"  tagged/auto 유효 {valid.mean():.3f}  전체 불일치율 {tm['disagree'].mean(skipna=True):.4f}")

log("(투수, 시즌) 프로파일 집계...")
grp = tm.groupby(["pitcher_id", "season"])
prof = grp.agg(
    n=("rel_speed", "size"),
    disagree=("disagree", "mean"),
    velo_loss=("velo_loss", "mean"),
    velo_loss_sd=("velo_loss", "std"),
    velo_retain=("velo_retain", "mean"),
)
# 압박 상황 불일치 - 평소 불일치
d3 = tm[tm.is_3ball == 1].groupby(["pitcher_id", "season"])["disagree"].mean().rename("disagree_3ball")
dd = tm[tm.deep_pa == 1].groupby(["pitcher_id", "season"])["disagree"].mean().rename("disagree_deeppa")
prof = prof.join(d3, how="left").join(dd, how="left")
prof["disagree_press"] = prof["disagree_3ball"] - prof["disagree"]
prof["disagree_deep"] = prof["disagree_deeppa"] - prof["disagree"]

# auto 구종 레퍼토리 엔트로피 (실제로 나온 구질 기준)
mix = tm.groupby(["pitcher_id", "season", "auto_pitch_type"]).size().rename("c").reset_index()
tot = mix.groupby(["pitcher_id", "season"])["c"].transform("sum")
mix["p"] = mix["c"] / tot
ent = mix.assign(e=-mix["p"] * np.log(mix["p"].clip(1e-9))).groupby(["pitcher_id", "season"])["e"].sum()
ntypes = mix.groupby(["pitcher_id", "season"]).size().rename("n_types")
prof = prof.join(ent.rename("auto_entropy"), how="left").join(ntypes, how="left")

CAND = ["disagree", "disagree_press", "disagree_deep", "velo_loss", "velo_loss_sd",
        "velo_retain", "auto_entropy", "n_types"]
log(f"  프로파일 {len(prof):,}개 (투수x시즌), 후보 {len(CAND)}개")

log("누적(<=season-1) 확장 후 2024 행에 매핑...")
seasons_range = sorted(df["season"].unique().tolist())
prof = prof.reset_index()
# 투구수 가중 누적평균
rows = []
for pid, gsub in prof.groupby("pitcher_id", sort=False):
    gsub = gsub.sort_values("season")
    cn = 0.0
    acc = {c: 0.0 for c in CAND}
    for _, r in gsub.iterrows():
        nn = float(r["n"])
        for c in CAND:
            v = r[c]
            if np.isfinite(v):
                acc[c] += v * nn
        cn += nn
        rows.append({"pitcher_id": pid, "season": int(r["season"]), "cum_n": cn,
                     **{c: (acc[c] / cn if cn > 0 else np.nan) for c in CAND}})
cum = pd.DataFrame(rows)
piv = {c: cum.pivot(index="pitcher_id", columns="season", values=c)
              .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True) for c in CAND + ["cum_n"]}

d_all = pd.read_csv("../data/train.csv", encoding="utf-8-sig", usecols=["season", "pitcher_id"])
idx = pd.MultiIndex.from_arrays([d_all["pitcher_id"], d_all["season"] - 1])
feats = {}
K = 200.0
n_cell = np.nan_to_num(piv["cum_n"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
for c in CAND:
    v = piv[c].reindex(idx).to_numpy().astype(np.float64)
    gm = np.nanmedian(v)
    feats[c] = (n_cell * np.nan_to_num(v, nan=gm) + K * gm) / (n_cell + K)
feats["tm2_n"] = np.log1p(n_cell)
log(f"  매칭율 {(n_cell>0).mean():.3f}")

log("스크리닝...")
rows = []
for k, v in feats.items():
    gn, pc = partial_gain(y_va, p_gbdt, np.asarray(v)[va])
    rows.append(dict(feature=k, gain=gn, sigma=abs(pc) * np.sqrt(len(y_va)), sign=np.sign(pc)))
res = pd.DataFrame(rows).sort_values("gain", ascending=False)
print()
print("=" * 60)
print(f"{'피처':<22}{'증분점수':>12}{'시그마':>9}{'부호':>7}")
print("-" * 60)
for _, r in res.iterrows():
    print(f"{r.feature:<22}{r.gain:12.2f}{r.sigma:9.1f}{r['sign']:+7.0f}")
Z = np.column_stack([np.asarray(feats[c])[va] for c in feats])
g, kk = block_gain(y_va, p_gbdt, Z)
print(f"\n[전체 블록] 증분 {g:+.2f}점 (k={kk})")
res.to_csv("phase82_tm_intent.csv", index=False)
log(f"총 {time.time()-t0:.0f}s")
