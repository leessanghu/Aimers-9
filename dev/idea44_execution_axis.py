"""idea44 — 투구 실행(execution) 물리축 aux head. 완전히 새로운 정보 원천.

## 왜 새로운가
지금까지 aux축 5종(middle/ball/strike/other/reverse)은 전부 `asof_pitcher_*_rate`
누적컬럼 차분이라는 **동일 원천**에서 나왔다. 그래서 수확체감이 뚜렷했다
(초기 4건 평균 +7.49 -> 최근 3건 평균 +1.76).
이 축은 trackman 물리계측이라는 **다른 원천**이다.

## 검증된 신호 (715,189행 매칭, 전체의 48.5%)
구종별 성공률: fastball 0.5428 / offspeed 0.5138 / breaking 0.4839 (5.9%p)
구속편차 5분위 성공률 격차: 구종통제 전 4.09%p -> 후 2.87%p (70% 생존)
IVB편차 격차 2.23%p / 회전수편차 격차 2.14%p (구종통제 후)

## 왜 피처가 아니라 aux head인가
"이번 투구에서 무슨 구종을 어떤 구속으로 던질지"는 추론시점에 알 수 없다.
게다가 trackman은 2019~2024만 있고 2025(테스트)가 없다.
-> 피처로는 원리적으로 불가. 그러나 **train 전용 aux head 타겟**으로는 사용 가능하고,
   추론시엔 head0(y)만 쓰므로 2025 trackman이 필요 없다. midaxis(v50)와 동일 구조.

## head 구성 (전부 '높을수록 제구에 유리' 방향으로 정렬)
head0 = y (control_success)
head1 = is_fastball          (제구가 가장 쉬운 구종인가)
head2 = 1 - 구속편차 백분위   (구종내 정상 구속으로 던졌는가)
head3 = 1 - IVB편차 백분위    (구종내 정상 무브먼트로 던졌는가)
매칭 안 된 51.5%는 NaN -> MultiRMSEWithMissingValues가 해당 head만 자동 제외.

## Rule.md §4 준수
trackman은 주최측 제공 데이터이고, train 행에만 셀 단위로 매칭한다.
test 행 간 참조 없음. 추론은 head0만 사용.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea44_cache"
os.makedirs(CD, exist_ok=True)
MATCH_CACHE = "tm_exec_matched.parquet"
t0 = time.time()
SEEDS = [42, 7]
KEY = ["season", "game_month", "game_dayofweek", "inning", "top_bottom",
       "balls_before", "strikes_before", "outs_before", "tm_id"]


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

if os.path.exists(MATCH_CACHE):
    heads = pd.read_parquet(MATCH_CACHE)
    log(f"  매칭캐시 재사용 {MATCH_CACHE}")
else:
    log("trackman 셀 매칭 (유일셀만)...")
    pm = pd.read_csv("pitcher_map.csv").sort_values("sim", ascending=False).drop_duplicates("tm_id")
    p2t = pm.set_index("pitcher_id")["tm_id"]
    tr = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                     usecols=["row_id", "season", "game_month", "game_dayofweek", "inning",
                              "top_bottom", "balls_before", "strikes_before", "outs_before",
                              "pitcher_id"])
    tr["row_num"] = tr["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    tr = tr.sort_values("row_num").reset_index(drop=True)
    assert (tr["row_num"].to_numpy() == meta["row_num"].to_numpy()).all(), "행 정렬 불일치"
    tr["tm_id"] = tr["pitcher_id"].map(p2t)
    tr["top_bottom"] = tr["top_bottom"].map(lambda v: {"T": "Top", "B": "Bottom"}.get(v, v))

    tm = pd.read_csv("../data/trackman_history.csv", encoding="utf-8-sig",
                     usecols=KEY[:-1] + ["pitcher_trackman_id", "rel_speed",
                                         "induced_vert_break", "pitch_type_group"])
    tm = tm.rename(columns={"pitcher_trackman_id": "tm_id"})
    tm["top_bottom"] = tm["top_bottom"].astype(str)
    grp = tm.groupby(KEY)
    agg = grp[["rel_speed", "induced_vert_break", "pitch_type_group"]].first()
    agg = agg.join(grp.size().rename("_n"))
    agg = agg[agg["_n"] == 1].drop(columns="_n")

    j = tr.join(agg, on=KEY, how="left")
    ok = j["rel_speed"].notna()
    log(f"  매칭 {ok.sum():,}/{len(j):,} ({ok.mean()*100:.1f}%)")

    # 구종 내부 편차 (구종효과 제거) -> 백분위로 정규화
    gk = ["pitcher_id", "season", "pitch_type_group"]
    dev_sp = (j["rel_speed"] - j.groupby(gk)["rel_speed"].transform("median")).abs()
    dev_vb = (j["induced_vert_break"] - j.groupby(gk)["induced_vert_break"].transform("median")).abs()

    heads = pd.DataFrame(index=j.index)
    heads["h_fastball"] = np.where(ok, (j["pitch_type_group"] == "fastball").astype(np.float64), np.nan)
    heads["h_speed_ok"] = np.where(ok, 1.0 - dev_sp.rank(pct=True), np.nan)
    heads["h_break_ok"] = np.where(ok, 1.0 - dev_vb.rank(pct=True), np.nan)
    heads.to_parquet(MATCH_CACHE)
    log(f"  저장 {MATCH_CACHE}")

for c in heads.columns:
    v = heads[c]
    log(f"  {c}: 유효 {v.notna().mean()*100:.1f}%  mean={v.mean():.4f}")

CAT_PARAMS = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
                  loss_function="MultiRMSEWithMissingValues", early_stopping_rounds=50)

VARIANTS = {
    "exec3": ["h_fastball", "h_speed_ok", "h_break_ok"],
    "ptype_only": ["h_fastball"],
    "exec_only": ["h_speed_ok", "h_break_ok"],
}

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def sc(p):
        return 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)

    base3 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) * np.load(f"phase90_cache/{tag}_snc_{n}.npy")
                   for n in ["d6", "d8"]], axis=0)
    mr = np.mean([np.load(f"idea13_cache/{tag}_multires_s{k}.npy") for k in [42, 7]], axis=0)
    od = np.mean([np.load(f"idea13_cache/{tag}_ordinal_s{k}.npy") for k in [42, 7]], axis=0)
    md = np.mean([np.load(f"idea31_cache/{tag}_midaxis_s{k}.npy") for k in [42, 7]], axis=0)
    v50l = 0.20 * base3 + 0.40 * hur + 0.10 * mr + 0.20 * od + 0.10 * md
    log(f"  v50local={sc(v50l):.2f}")

    for vname, cols in VARIANTS.items():
        Ymat = np.column_stack([y] + [heads[c].to_numpy() for c in cols])
        ps = []
        for seed in SEEDS:
            f = f"{CD}/{tag}_{vname}_s{seed}.npy"
            if os.path.exists(f):
                ps.append(np.load(f)); continue
            ts = time.time()
            n_es = int(tr_m.sum() * 0.92)
            mdl = CatBoostRegressor(**CAT_PARAMS, random_seed=seed)
            mdl.fit(X.loc[tr_m].iloc[:n_es], Ymat[tr_m][:n_es], sample_weight=w[tr_m][:n_es],
                    eval_set=(X.loc[tr_m].iloc[n_es:], Ymat[tr_m][n_es:]))
            p = np.clip(mdl.predict(X.loc[va_m]), 0.0, 1.0)[:, 0]
            np.save(f, p); ps.append(p)
            log(f"    [{tag}/{vname}/s{seed}] best_iter={mdl.best_iteration_} ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")
        avg = np.mean(ps, axis=0)
        spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
        row = {"solo": sc(avg), "spread": spread}
        for wv in [0.10, 0.15]:
            blend = (1 - wv) * v50l + wv * avg
            row[f"w{wv}"] = sc(blend) - sc(v50l)
        results[(tag, vname)] = row
        log(f"  {tag}/{vname:<11} 단독={row['solo']:8.2f} 시드폭={spread:6.2f} "
            f"w0.10={row['w0.1']:+.2f} w0.15={row['w0.15']:+.2f}")

print()
print("=" * 78)
print(f"{'fold':<6}{'변형':<13}{'단독':>10}{'시드폭':>9}{'w=0.10':>10}{'w=0.15':>10}")
for (tag, vn), r in results.items():
    print(f"{tag:<6}{vn:<13}{r['solo']:10.2f}{r['spread']:9.2f}{r['w0.1']:+10.2f}{r['w0.15']:+10.2f}")
print()
print("기준: middle축 fold A w0.1=+1.08 -> 실측 +7.72 / other축 fold A -0.19 -> 실측 +3.25")
print("aux head는 fold A가 과소평가하는 계열(편향 +3~6). fold A -2 이상이면 실측 가치 있음.")
log(f"총 {time.time()-t0:.0f}s")
