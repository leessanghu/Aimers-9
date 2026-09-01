"""아이디어 7 — Blockwise Target Dropout Ensemble (Codex 제안).

배경: CatBoost seed를 바꿔도 예측 상관이 0.97로 안 내려가는 이유는 모든 모델이 같은
147만개 label gradient를 본다는 것. 행 단위 무작위 subsampling은 인접 투구가 거의
같은 정보라 gradient가 별로 안 바뀐다. 대신 '등판(appearance)' 전체를 블록으로 지운다
-> 특정 경기의 우연한 성공/실패 연속이 split을 지배하지 못하게, 모델마다 다른 투수
상태/경기 패턴을 놓치게 만든다 (seed/rsm보다 강한 구조적 다양성).

파일럿(제안대로): HGB d6 하나로 3-mask, 각각 20% 드롭. 상관/한계이득 확인 후
5개로 확장할지 결정.

등판 경계는 formfeat.py의 build_role_table과 동일한 근사 로직 재사용
(같은 투수의 (season,month,dayofweek) 바뀌거나 inning이 감소하면 새 등판).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

CD = "idea7_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
N_MASK = 3
DROP_FRAC = 0.20


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)

n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
ordr_pid = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
pid_o = meta["pitcher_id"].to_numpy()[ordr_pid]
step = np.zeros(len(meta), dtype=bool)
step[ordr_pid[:-1]] = (pid_o[1:] == pid_o[:-1]) & (np.diff(n_[ordr_pid]) == 1)

log("등판(appearance) 블록 복원 (formfeat.py와 동일 로직)...")
full = pd.read_csv("../data/train.csv", encoding="utf-8-sig",
                   usecols=["row_id", "pitcher_id", "season", "game_month", "game_dayofweek", "inning"])
full["row_num"] = full["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
d = full.sort_values(["pitcher_id", "row_num"]).reset_index()  # index=원본 meta 순서

pid_arr = d["pitcher_id"].to_numpy()
key = (d["season"].to_numpy() * 10000 + d["game_month"].to_numpy() * 100 + d["game_dayofweek"].to_numpy())
inn = d["inning"].to_numpy()
new_p = np.empty(len(d), dtype=bool); new_p[0] = True; new_p[1:] = pid_arr[1:] != pid_arr[:-1]
new_k = np.empty(len(d), dtype=bool); new_k[0] = True; new_k[1:] = key[1:] != key[:-1]
drop_ = np.empty(len(d), dtype=bool); drop_[0] = False; drop_[1:] = inn[1:] < inn[:-1]
app_id_sorted = np.cumsum(new_p | new_k | drop_)

app_id = np.empty(len(meta), dtype=np.int64)
app_id[d["index"].to_numpy()] = app_id_sorted
n_app = app_id.max() + 1
log(f"  등판 {n_app:,}개 복원 (평균 {len(meta)/n_app:.1f}구/등판)")

rng = np.random.RandomState(42)
app_bucket_of = rng.randint(0, N_MASK, size=n_app)  # 등판 -> 버킷 (0..N_MASK-1)
row_bucket = app_bucket_of[app_id]  # 행 -> 그 행이 속한 등판의 버킷


def drop_mask_for(mask_id):
    """이 mask_id 모델이 제외할 행(=해당 버킷의 등판 전체)."""
    return row_bucket == mask_id


HGB_D6 = dict(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03, l2_regularization=5.0,
             early_stopping=True, validation_fraction=0.1, n_iter_no_change=20, random_state=42)

results = {}
for upto, val, tag in [(2023, 2024, "A"), (2021, 2022, "C"), (2022, 2023, "B")]:
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr_m = (seasons <= upto) & step
    va_m = seasons == val
    yv = y[va_m]
    r = yv.mean(); BS = r * (1 - r)
    w = 0.5 ** ((upto - seasons) / 2.0)

    def score(p):
        return 1e5 * (1 - np.mean((p - yv) ** 2) / BS)

    base_full = np.load(f"phase90_cache/{tag}_base_d6.npy")  # 전체데이터 d6 (동일 config)

    mask_preds = []
    for mid in range(N_MASK):
        f = f"{CD}/{tag}_mask{mid}.npy"
        if os.path.exists(f):
            mask_preds.append(np.load(f))
            continue
        excl = drop_mask_for(mid)
        tr_sub = tr_m & (~excl)
        ts = time.time()
        m = HistGradientBoostingClassifier(**HGB_D6).fit(X.loc[tr_sub], y[tr_sub], sample_weight=w[tr_sub])
        p = m.predict_proba(X.loc[va_m])[:, 1]
        np.save(f, p)
        mask_preds.append(p)
        log(f"  mask{mid} 완료 iters={m.n_iter_} 학습행={tr_sub.sum():,}(제외 {excl[tr_m].mean()*100:.1f}%)"
           f" ({time.time()-ts:.0f}s)")
    p_ens = np.mean(mask_preds, axis=0)

    s_full = score(base_full)
    s_ens = score(p_ens)
    corr_pairs = [np.corrcoef(mask_preds[i], mask_preds[j])[0, 1]
                 for i in range(N_MASK) for j in range(i + 1, N_MASK)]
    log(f"  full데이터 d6={s_full:.2f}  {N_MASK}-mask평균={s_ens:.2f}  (대비 {s_ens-s_full:+.2f})"
       f"  마스크간 평균상관={np.mean(corr_pairs):.4f}")

    v29 = np.mean([np.load(f"phase90_cache/{tag}_base_{n}.npy") for n in ["d6", "d8", "sub"]], axis=0)
    s_v29 = score(v29)
    row = dict(full_d6=s_full, mask_ens=s_ens, corr=np.mean(corr_pairs), v29local=s_v29)
    for wv in [0.15, 0.3, 0.5]:
        blend = (1 - wv) * v29 + wv * p_ens
        row[f"w{wv}"] = score(blend)
        log(f"  v29local+mask_ens(w={wv}) = {row[f'w{wv}']:.2f}  (v29local대비 {row[f'w{wv}']-s_v29:+.2f})")
    results[tag] = row

print()
print("=" * 90)
hdr = f"{'fold':<6}{'full_d6':>10}{'mask_ens':>10}{'ens대비':>9}{'상관':>8}{'v29local':>10}"
for w in [0.15, 0.3, 0.5]:
    hdr += f"{'w='+str(w):>10}"
print(hdr)
for tag, r in results.items():
    row = f"{tag:<6}{r['full_d6']:10.2f}{r['mask_ens']:10.2f}{r['mask_ens']-r['full_d6']:+9.2f}" \
         f"{r['corr']:8.4f}{r['v29local']:10.2f}"
    for w in [0.15, 0.3, 0.5]:
        row += f"{r[f'w{w}']:10.2f}"
    print(row)

print()
for wv in [0.15, 0.3, 0.5]:
    gains = [results[t][f"w{wv}"] - results[t]["v29local"] for t in ["A", "C", "B"]]
    print(f"w={wv}: 폴드별 이득 {[round(g,2) for g in gains]}  최소={min(gains):+.2f}  "
         f"{'채택검토(5-mask로 확장)' if min(gains) > 2 else '기각'}")
pd.DataFrame(results).T.to_csv("idea7_results.csv")
log(f"총 {time.time()-t0:.0f}s")
