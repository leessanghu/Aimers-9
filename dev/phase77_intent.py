"""phase77 — '의도(intent)' 축 스크리닝.

동기:
    control_success는 '의도한 곳에 갔는가'인데 의도 자체는 데이터에 없다.
    그런데 복원한 행 단위 라벨(일치율 1.000000, 커버리지 99.95%)의 실패 모드 구성비가
    의도를 역으로 알려준다:
        실패가 주로 reverse -> 코너를 노렸다 (빗나가면 반대쪽)
        실패가 주로 safe ball -> 애초에 존 밖을 노렸다 (유인구)
        실패가 주로 middle -> 코너 노렸는데 가운데로 샜다
    즉 실패 모드 구성비 = '의도 공격성'이라는 스킬과 별개인 잠재변수의 관측치.

    SHAP 비대칭이 이를 뒷받침한다:
        asof_pitcher_ball_rate_smooth      7위  0.01729   <- 존 밖 빈도 = 의도 대리지표
        asof_pitcher_reverse_rate_smooth  37위  0.00252
        asof_pitcher_strike_rate_smooth   91위  0.00046
        asof_pitcher_middle_rate_smooth  113위  0.00033
    원시 수준은 전체 스킬과 뒤엉켜 있어 트리가 못 쓴다. 실패 중 '비중'으로 정규화해야
    스킬과 분리된다. 그리고 트리는 비율을 split으로 근사하기 어렵다.

    결정적으로, 지금 모델이 가진 건 주변확률뿐이다(success/reverse/middle/ball/strike 각각).
    복원 라벨로는 결합확률을 만들 수 있는데 이건 모델에 전혀 없다:
        P(strike|success)  의도대로 갔을 때 존 안이었나  = 공격적 의도 vs 안전한 의도
        P(ball|fail)       빗나갈 때 안전하게 빠지나
        P(middle|fail)     빗나갈 때 위험하게 몰리나

방법: phase64b와 동일 (p 통제 부분상관, 자유도 1, 귀무편향 0.4점).
      해석: 4점=1시그마, 16점=2시그마, 36점=3시그마 (n=253,507)

규칙: A/B는 행 자신의 공식 asof_* 컬럼만 사용(행 내부). C는 (투수, season-1) 룩업으로
      train에서만 만든 정적 테이블 — inseason/platoon과 완전히 동일한 패턴.
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

VALID_SEASON = 2024
t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


def _clean(z):
    z = np.asarray(z, dtype=np.float64).copy()
    bad = ~np.isfinite(z)
    if bad.any():
        z[bad] = np.nanmedian(z[~bad]) if (~bad).any() else 0.0
    return z


def partial_gain(y, p, z):
    z = _clean(z)
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
    Z = np.column_stack([_clean(Z[:, j]) for j in range(Z.shape[1])])
    Z = Z[:, [j for j in range(Z.shape[1]) if Z[:, j].std() > 0]]
    n, k = len(y), Z.shape[1]
    X0 = np.column_stack([np.ones(n), p])
    X1 = np.column_stack([X0, Z])

    def r2(X):
        c = np.linalg.lstsq(X, y, rcond=None)[0]
        return 1 - (y - X @ c).var() / y.var()

    return 1e5 * (r2(X1) - r2(X0) - k / n), k


log("로드...")
COLS = ["row_id", "season", "pitcher_id", "control_success", "asof_pitcher_n",
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_ball_rate", "asof_pitcher_strike_rate"]
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig", usecols=COLS)
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)

z = np.load("phase67_cache/phase69_preds.npz")
p_gbdt = 0.5 * z["hgb"] + 0.5 * z["cat3"]
y_all = df["control_success"].to_numpy(np.float64)

# 원 순서 기준 2024 마스크 (캐시 예측과 정렬 일치)
va_m = (df["season"] == VALID_SEASON).to_numpy()
y_va = y_all[va_m]
assert len(y_va) == len(p_gbdt), (len(y_va), len(p_gbdt))
log(f"valid={va_m.sum():,}  기준 잠재력={1e5*np.corrcoef(p_gbdt,y_va)[0,1]**2:.1f}")

# ----------------------------------------------------------------------
# A/B: 행 내부 비율 (원 순서 유지)
# ----------------------------------------------------------------------
n_ = df["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
S_ = df["asof_pitcher_success_rate"].to_numpy(np.float64)
R_ = df["asof_pitcher_reverse_rate"].to_numpy(np.float64)
M_ = df["asof_pitcher_middle_rate"].to_numpy(np.float64)
B_ = df["asof_pitcher_ball_rate"].to_numpy(np.float64)
K_ = df["asof_pitcher_strike_rate"].to_numpy(np.float64)
fail = np.clip(1.0 - S_, 1e-6, None)

cand = {}
# A. 실패 모드 구성비 (스킬과 분리된 '의도 공격성')
cand["rev_share"] = R_ / fail
cand["mid_share"] = M_ / fail
cand["safe_share"] = 1.0 - (R_ + M_) / fail
cand["rev_minus_mid_share"] = (R_ - M_) / fail
# B. 존/유인 의도 축
cand["ball_share_of_fail"] = B_ / fail
cand["chase_intent"] = B_ - fail            # 실패로 설명되는 것 이상의 볼 = 의도적 유인
cand["inplay_rate"] = 1.0 - B_ - K_
cand["zone_minus_success"] = K_ - S_        # 존에 넣는 빈도 vs 의도 적중 빈도
# 참조: 이미 모델이 강하게 쓰는 것 (0에 가까워야 정상)
cand["_ref_ball_rate"] = B_
cand["_ref_success_rate"] = S_

# ----------------------------------------------------------------------
# C: 결합확률 조건부 (복원 라벨 -> (투수, season-1) 누적 테이블)
# ----------------------------------------------------------------------
log("행 단위 라벨 복원...")
d = df.sort_values(["pitcher_id", "row_num"])
oi = d.index.to_numpy()           # 원 인덱스
nn = d["asof_pitcher_n"].fillna(0).to_numpy(np.float64)


def cum(col):
    return np.round(d[col].fillna(0).to_numpy(np.float64) * nn)


cS, cR, cM, cB, cK = [cum(c) for c in ["asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
                                       "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
                                       "asof_pitcher_strike_rate"]]
pid = d["pitcher_id"].to_numpy()
step = (pid[1:] == pid[:-1]) & (np.diff(nn) == 1)
sel = np.where(step)[0]
lab = pd.DataFrame({
    "orig": oi[:-1][step], "pid": pid[:-1][step], "sea": d["season"].to_numpy()[:-1][step],
    "s": np.diff(cS)[step], "r": np.diff(cR)[step], "m": np.diff(cM)[step],
    "b": np.diff(cB)[step], "k": np.diff(cK)[step]})
log(f"  복원 {len(lab):,}행 ({100*len(lab)/len(df):.2f}%)  "
    f"일치율 {(lab.s.to_numpy()==y_all[lab.orig.to_numpy()]).mean():.6f}")

lab["f"] = 1 - lab.s
lab["inplay"] = 1 - lab.b - lab.k
# 결합 카운트
lab["s_k"] = lab.s * lab.k          # 성공 & 스트라이크  -> 공격적 의도 적중
lab["s_b"] = lab.s * lab.b          # 성공 & 볼          -> 유인구 의도 적중
lab["s_ip"] = lab.s * lab.inplay
lab["f_b"] = lab.f * lab.b          # 실패 & 볼          -> 안전한 빗나감
lab["f_k"] = lab.f * lab.k
lab["f_r"] = lab.f * (lab.r > 0)
lab["f_m"] = lab.f * (lab.m > 0)

agg = lab.groupby(["pid", "sea"])[["s", "f", "s_k", "s_b", "s_ip", "f_b", "f_k", "f_r", "f_m"]].sum()
agg["n"] = lab.groupby(["pid", "sea"]).size()
cs = agg.sort_index().groupby(level=0).cumsum()

seasons_range = sorted(df["season"].unique().tolist())
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])


def lk(col):
    pv = cs[col].unstack().reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    return np.nan_to_num(pv.reindex(idx).to_numpy().astype(np.float64), nan=0.0)


parts = {c: lk(c) for c in ["s", "f", "s_k", "s_b", "s_ip", "f_b", "f_k", "f_r", "f_m"]}
KSM = 200.0
gl = {c: float(lab[c].sum() / len(lab)) for c in ["s", "f"]}


def cond(num, den, gnum, gden):
    """P(num|den)을 전역값으로 축소. den이 작으면 전역으로 수렴."""
    gr = gnum / max(gden, 1e-9)
    return (parts[num] + KSM * gr) / (parts[den] + KSM)


gs = {c: float(lab[c].sum()) for c in lab.columns if c in
      ["s", "f", "s_k", "s_b", "s_ip", "f_b", "f_k", "f_r", "f_m"]}
cand["c_strike_given_succ"] = cond("s_k", "s", gs["s_k"], gs["s"])
cand["c_ball_given_succ"] = cond("s_b", "s", gs["s_b"], gs["s"])
cand["c_inplay_given_succ"] = cond("s_ip", "s", gs["s_ip"], gs["s"])
cand["c_ball_given_fail"] = cond("f_b", "f", gs["f_b"], gs["f"])
cand["c_rev_given_fail"] = cond("f_r", "f", gs["f_r"], gs["f"])
cand["c_mid_given_fail"] = cond("f_m", "f", gs["f_m"], gs["f"])
cand["c_aggress"] = cand["c_strike_given_succ"] - cand["c_ball_given_succ"]
cand["c_safe_miss"] = cand["c_ball_given_fail"] - cand["c_mid_given_fail"]

log("스크리닝...")
rows = []
for k, v in cand.items():
    gnum, pc = partial_gain(y_va, p_gbdt, np.asarray(v)[va_m])
    rows.append(dict(feature=k, gain=gnum, sigma=abs(pc) * np.sqrt(len(y_va)), pc=pc))
res = pd.DataFrame(rows).sort_values("gain", ascending=False)

print()
print("=" * 66)
print(f"{'피처':<26}{'증분점수':>10}{'시그마':>9}{'부호':>7}")
print("-" * 66)
for _, r in res.iterrows():
    mark = "  <- 참조(0이어야 정상)" if r.feature.startswith("_ref") else ""
    print(f"{r.feature:<26}{r.gain:10.2f}{r.sigma:9.1f}{np.sign(r.pc):+7.0f}{mark}")

for nm, keys in [("A 실패구성비", ["rev_share", "mid_share", "safe_share", "rev_minus_mid_share"]),
                 ("B 존/유인의도", ["ball_share_of_fail", "chase_intent", "inplay_rate", "zone_minus_success"]),
                 ("C 결합조건부", ["c_strike_given_succ", "c_ball_given_succ", "c_inplay_given_succ",
                                   "c_ball_given_fail", "c_rev_given_fail", "c_mid_given_fail",
                                   "c_aggress", "c_safe_miss"])]:
    Z = np.column_stack([np.asarray(cand[k])[va_m] for k in keys])
    g, kk = block_gain(y_va, p_gbdt, Z)
    print(f"\n[{nm}] 블록 증분 {g:+.2f}점 (k={kk})")

allk = [k for k in cand if not k.startswith("_ref")]
Z = np.column_stack([np.asarray(cand[k])[va_m] for k in allk])
g, kk = block_gain(y_va, p_gbdt, Z)
print(f"\n[전체 의도블록] 증분 {g:+.2f}점 (k={kk})")
res.to_csv("phase77_intent.csv", index=False)
log(f"저장 완료  총 {time.time()-t0:.0f}s")
