"""idea57 — 투수능력 잔차축(pitcher_resid). PCA붕괴 진단(PC1 94.89%=투수실력)에서
도출: 기존 aux head(middle/other/ball/count잔차/미래50투구)는 전부 y를 '다른 각도로
재정의'했을 뿐이라 결국 같은 실력축으로 수렴했다(N_eff≈1, r=0.98).

이번 축은 방향이 반대다: 타겟에서 투수실력(h1=투수-시즌 LOO 성공률, multires와 동일
공식 K_PS=15)을 명시적으로 빼서 head1 = y - h1 로 준다. 트리가 이 head도 동시에
설명해야 하므로 순수 실력 split이 아니라 '그 투수가 평소보다 잘한/못한 상황' split을
찾도록 규제된다.

로컬 델타는 판정에 안 쓴다(count_resid/cond_ball/future50 3/3이 로컬-실측 부호까지
반대였음, 실험 판정 규칙 3항). 대신 이 스크립트는 구조적 진단에 집중한다:
head0(direct) 예측이 x_ability_here/inseason_success_smooth(PC1의 핵심 성분)와
실제로 count_resid보다 덜 상관되는지 확인한다. 상관이 안 떨어지면 가설 기각.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

CD = "idea57_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
SEEDS = [42, 7]
K_PS = 15.0


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
bb = meta["balls_before"].to_numpy(np.float64)
st = meta["strikes_before"].to_numpy(np.float64)
count_state = bb * 4 + st

CAT = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function="MultiRMSE", early_stopping_rounds=50)
A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)

results = {}
diag = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = seasons <= upto
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)
    sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)
    g_tr = float(y[tr_m].mean())

    base = A([f"phase90_cache/{tag}_base_{n}.npy" for n in ["d6", "d8", "sub"]])
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) *
                   np.load(f"phase90_cache/{tag}_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
    mr = A([f"idea13_cache/{tag}_multires_s{k}.npy" for k in [42, 7]])
    od = A([f"idea13_cache/{tag}_ordinal_s{k}.npy" for k in [42, 7]])
    V47 = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
    B47 = sc(V47)
    log(f"  v47local={B47:.2f}")

    # 투수-시즌 LOO 성공률 h1 (multires v40과 동일 공식, train<=upto 내부에서만 사용
    # 가능하도록 fold별 재계산은 하지 않는다 -- multires도 train 전체에서 h1을 계산해
    # LOO 자체가 각 행 자신을 leave-out하므로 fold와 무관하게 안전하다. 단, train_m
    # 밖의 valid 행은 자기 자신의 season 합계에서 LOO한 값을 그대로 씀 -- 이는 v40/v62/
    # v63 프로덕션과 동일한 방식(추론시에도 valid=test 자신의 as-of 통계 사용).
    sub = pd.DataFrame({"pid": pid, "season": seasons, "y": y})
    ps_g = sub.groupby(["pid", "season"])["y"].agg(s="sum", n="count")
    sub = sub.join(ps_g, on=["pid", "season"])
    h1 = ((sub["s"] - sub["y"]) + K_PS * g_tr) / ((sub["n"] - 1) + K_PS)
    h1 = h1.to_numpy(np.float64)
    h_pitcher_resid = y - h1
    log(f"  h1(LOO 실력) mean={h1.mean():.4f} std={h1.std():.4f}  resid std={h_pitcher_resid.std():.4f}")

    # count_resid 비교대상 재현 (count prior, train<=upto 내부)
    ctab = pd.DataFrame({"cs": count_state[tr_m], "y": y[tr_m]}).groupby("cs")["y"].agg(["sum", "count"])
    K_C = 500.0
    ctab["prior"] = (ctab["sum"] + K_C * y[tr_m].mean()) / (ctab["count"] + K_C)
    cprior_all = pd.Series(count_state).map(ctab["prior"]).fillna(y[tr_m].mean()).to_numpy(np.float64)
    h_count_resid = y - cprior_all

    n_es = int(tr_m.sum() * 0.92)
    VARIANTS = {
        "pitcher_resid": [y, h_pitcher_resid],
        "count_resid_ref": [y, h_count_resid],
    }
    fold_preds = {}
    for name, heads in VARIANTS.items():
        Ymat = np.column_stack([np.asarray(h, dtype=np.float64) for h in heads])
        ps = []
        for seed in SEEDS:
            f = f"{CD}/{tag}_{name}_s{seed}.npy"
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
        fold_preds[name] = u
        spread = max(sc(p) for p in ps) - min(sc(p) for p in ps)
        d = sc(0.90 * V47 + 0.10 * u) - B47
        results.setdefault(name, {})[tag] = dict(solo=sc(u), spread=spread, delta=d)
        log(f"  {name:<16} 단독={sc(u):.2f} 시드폭={spread:.2f} Δ={d:+.2f}")

    # ---- 핵심 진단: PC1(투수실력) 성분과의 상관 ----
    ability_here = X.loc[va_m, "x_ability_here"].to_numpy(np.float64)
    inseason_succ = X.loc[va_m, "inseason_success_smooth"].to_numpy(np.float64)
    cmd_idx = X.loc[va_m, "inseason_cmd_index"].to_numpy(np.float64)
    for name, u in fold_preds.items():
        c1 = np.corrcoef(u, ability_here)[0, 1]
        c2 = np.corrcoef(u, inseason_succ)[0, 1]
        c3 = np.corrcoef(u, cmd_idx)[0, 1]
        c4 = np.corrcoef(u, V47)[0, 1]
        diag.setdefault(name, {})[tag] = dict(ability=c1, inseason=c2, cmd=c3, v47=c4)
        log(f"  [진단] {name:<16} corr(ability_here)={c1:.4f} corr(inseason_succ)={c2:.4f} "
            f"corr(cmd_idx)={c3:.4f} corr(V47)={c4:.4f}")

print()
print("=" * 78)
print("pitcher_resid vs count_resid: fold A/C 결과 + PC1(투수실력) 상관 진단")
print("=" * 78)
for name, r in results.items():
    a = r.get("A", {}); c = r.get("C", {})
    print(f"{name:<16} foldA_Δ={a.get('delta',0):+7.2f}(폭{a.get('spread',0):.2f})  "
          f"foldC_Δ={c.get('delta',0):+7.2f}(폭{c.get('spread',0):.2f})")
for name, d in diag.items():
    a = d.get("A", {}); c = d.get("C", {})
    print(f"{name:<16} foldA corr(ability/inseason/cmd/V47)="
          f"{a.get('ability',0):.4f}/{a.get('inseason',0):.4f}/{a.get('cmd',0):.4f}/{a.get('v47',0):.4f}  "
          f"foldC corr={c.get('ability',0):.4f}/{c.get('inseason',0):.4f}/{c.get('cmd',0):.4f}/{c.get('v47',0):.4f}")
print()
print("판정: pitcher_resid의 corr(ability_here/inseason_succ)가 count_resid_ref보다")
print("뚜렷이(0.03+) 낮으면 가설(실력축 분리 성공) 지지. 비슷하거나 높으면 기각.")
log(f"총 {time.time()-t0:.0f}s")
