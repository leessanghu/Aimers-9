"""v75 = v66 + MLP(pitcher/batter 16dim embedding) 블렌드. idea62 mse_emb8 레시피
(fold A: 단독=711, corr(v66)=0.88, 로컬최적w=0.000 -- 블렌드 손해로 로컬은 판정함).
그럼에도 실측 태우는 이유: 이 계열에서 로컬 판정이 반복적으로 틀렸었고(v62/63/64
로컬 음수->실측 3/3 양수), corr(v66)이 기존 축(0.98) 대비 압도적으로 낮은 유일한
후보라 실측 검증 가치가 있다.

torch로 학습하되 추론은 전부 numpy 행렬곱으로 이식한다(requirements.txt 불변,
환경 리스크 0 -- v72 RNG 사고 재발 방지).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import batter_split as bsplit
from batterform import K_BATTER, build_batter_table, transform_batter
from career_volatility import K_VOL, build_volatility_table, transform_volatility
from count_split import K_COUNT, build_count_table, transform_count
from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from formfeat import build_role_table, transform_form, transform_role
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from inseason_full import build_global_priors, build_season_end_table_full, transform_inseason_full
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
OUT_DIR = "../submit/model"
t0 = time.time()
EMB = 8
MLP_WEIGHT = 0.05
MIN_COUNT = 30
torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("v66 아티팩트 로드...")
v66 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v66.pkl"))
log(f"  hgbs={len(v66['hgbs'])} cats={len(v66['cats'])}")

log("데이터 로드 + 피처 재구성...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy(np.float32)
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())

fb = FeatureBuilder(seed=42, include_raw_rates=False, team_te_mode="expanding").fit(df)
X_base = fb.transform_train_oof(df).reset_index(drop=True)
se = build_season_end_table(df)
X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
X_plt = transform_platoon(df, build_platoon_table(df), prior, sr, k=K_PLATOON).reset_index(drop=True)
it, io = build_inning_table(df), build_inning_offset(df)
X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)
X_cnt = transform_count(df, build_count_table(df), prior, sr, k=K_COUNT).reset_index(drop=True)
X_pt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), prior, g, sr).reset_index(drop=True)
X_ly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0).reset_index(drop=True)
X_vol = transform_volatility(df, build_volatility_table(se), sr, k=K_VOL).reset_index(drop=True)
role_tbl = build_role_table(df)
X_role = transform_role(df, role_tbl, sr).reset_index(drop=True)
base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
X_form = transform_form(df, X_role, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                        base_middle).reset_index(drop=True)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
X_tm = transform_trackman(df, prof, sr).reset_index(drop=True)
lown_thr = float(df["asof_pitcher_n"].fillna(0).median())
X_tmx = add_lown_interactions(X_tm, df["asof_pitcher_n"].to_numpy(np.float64), lown_thr).reset_index(drop=True)
X_bat = transform_batter(df, build_batter_table(df), sr, g, k=K_BATTER).reset_index(drop=True)
n_end_row = np.nan_to_num(piv["N_end"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
X_isf = transform_inseason_full(df, build_season_end_table_full(df), build_global_priors(df), sr,
                                n_end_row, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                                X_ins["inseason_reverse_smooth"].to_numpy(np.float64)).reset_index(drop=True)
g_bmid = float(df["asof_batter_middle_rate"].mean(skipna=True))
X_bmid = bsplit.transform_batter_middle(df, bsplit.build_batter_middle_table(df), sr, g_bmid).reset_index(drop=True)
bmarg = bsplit.build_batter_marginal(df)
b_prior = bsplit.lookup_batter_prior(df, bmarg, sr, g)
X_bplat = bsplit.transform_bplatoon(df, bsplit.build_bplatoon_table(df), b_prior, sr,
                                    k=bsplit.K_BPLATOON).reset_index(drop=True)

X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
C = add_crosses(X)
X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm, X_tmx, X_bat,
               X_isf, X_bmid, X_bplat], axis=1).astype(np.float64)
X = X[v66["feature_order"]].astype(np.float32)
log(f"피처 {X.shape[1]}개")

pid = df["pitcher_id"].to_numpy()
bid = df["batter_id"].to_numpy()
season = df["season"].to_numpy(np.float64)
w_rec = (0.5 ** ((season.max() - season) / 2.0)).astype(np.float32)
Xn = X.to_numpy(np.float32)

vc_p = pd.Series(pid).value_counts()
vc_b = pd.Series(bid).value_counts()
pmap = {v: i + 1 for i, v in enumerate(vc_p[vc_p >= MIN_COUNT].index)}
bmap = {v: i + 1 for i, v in enumerate(vc_b[vc_b >= MIN_COUNT].index)}
ip = np.array([pmap.get(v, 0) for v in pid], dtype=np.int64)
ib = np.array([bmap.get(v, 0) for v in bid], dtype=np.int64)
log(f"pitcher vocab={len(pmap)+1} batter vocab={len(bmap)+1}")

mu = Xn.mean(0)
sd = Xn.std(0)
sd[sd < 1e-8] = 1.0
Z = np.clip((Xn - mu) / sd, -10, 10).astype(np.float32)

n = len(X)
rng = np.random.default_rng(0)
perm = rng.permutation(n)
cut = int(n * 0.92)
fit_i, es_i = perm[:cut], perm[cut:]
T = lambda a, i: torch.as_tensor(a[i])


class Net(nn.Module):
    def __init__(self, n_num, n_p, n_b):
        super().__init__()
        self.ep = nn.Embedding(n_p, EMB)
        self.eb = nn.Embedding(n_b, EMB)
        nn.init.normal_(self.ep.weight, 0, 0.01)
        nn.init.normal_(self.eb.weight, 0, 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(n_num + 2 * EMB, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 1),
        )

    def forward(self, xn, xp, xb):
        return self.mlp(torch.cat([xn, self.ep(xp), self.eb(xb)], 1)).squeeze(1)


log("MLP 학습 (전체데이터, mse_emb8 레시피)...")
torch.manual_seed(42)
net = Net(Xn.shape[1], len(pmap) + 1, len(bmap) + 1)
opt = torch.optim.AdamW([
    {"params": list(net.mlp.parameters()), "weight_decay": 1e-5},
    {"params": list(net.ep.parameters()) + list(net.eb.parameters()), "weight_decay": 1e-3},
], lr=5e-4)
MAX_EP, PAT = 40, 6
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EP)
ez, ep_, eb_, ey, ew = T(Z, es_i), T(ip, es_i), T(ib, es_i), T(y, es_i), T(w_rec, es_i)
best, best_state, bad = np.inf, None, 0
for e in range(MAX_EP):
    net.train()
    p_ep = np.random.default_rng(42 + e).permutation(len(fit_i))
    ts = time.time()
    for s in range(0, len(p_ep), 8192):
        j = fit_i[p_ep[s:s + 8192]]
        opt.zero_grad()
        w = T(w_rec, j)
        loss = (((torch.sigmoid(net(T(Z, j), T(ip, j), T(ib, j))) - T(y, j)) ** 2) * w).sum() / w.sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
        opt.step()
    sched.step()
    net.eval()
    with torch.no_grad():
        vl = float((((torch.sigmoid(net(ez, ep_, eb_)) - ey) ** 2) * ew).sum() / ew.sum())
    log(f"  epoch {e:2d} es={vl:.6f} ({time.time()-ts:.0f}s)")
    if vl < best - 1e-7:
        best, bad = vl, 0
        best_state = {k: v.clone() for k, v in net.state_dict().items()}
    else:
        bad += 1
        if bad >= PAT:
            log(f"  early stop @ epoch {e}")
            break
net.load_state_dict(best_state)
log(f"학습완료 best_es={best:.6f}")

# ---- numpy 가중치 추출 (torch 의존성 없이 추론) ----
sd_ = net.state_dict()
mlp_weights = {
    "emb_p": sd_["ep.weight"].numpy().astype(np.float32),
    "emb_b": sd_["eb.weight"].numpy().astype(np.float32),
    "W1": sd_["mlp.0.weight"].numpy().astype(np.float32).T, "b1": sd_["mlp.0.bias"].numpy().astype(np.float32),
    "W2": sd_["mlp.3.weight"].numpy().astype(np.float32).T, "b2": sd_["mlp.3.bias"].numpy().astype(np.float32),
    "W3": sd_["mlp.6.weight"].numpy().astype(np.float32).T, "b3": sd_["mlp.6.bias"].numpy().astype(np.float32),
    "mu": mu.astype(np.float32), "sd": sd.astype(np.float32),
    "pmap": pmap, "bmap": bmap,
}

# 자체 검증: numpy 순전파가 torch와 일치하는지 확인
def np_forward(Xrow, ip_row, ib_row, w):
    z = np.clip((Xrow - w["mu"]) / w["sd"], -10, 10)
    ep_v = w["emb_p"][ip_row]
    eb_v = w["emb_b"][ib_row]
    h = np.concatenate([z, ep_v, eb_v], axis=1)
    h = np.maximum(h @ w["W1"] + w["b1"], 0)
    h = np.maximum(h @ w["W2"] + w["b2"], 0)
    out = (h @ w["W3"] + w["b3"]).squeeze(1)
    return 1.0 / (1.0 + np.exp(-out))

chk_i = es_i[:2000]
p_np = np_forward(Xn[chk_i], ip[chk_i], ib[chk_i], mlp_weights)
net.eval()
with torch.no_grad():
    p_torch = torch.sigmoid(net(T(Z, chk_i), T(ip, chk_i), T(ib, chk_i))).numpy()
max_diff = np.abs(p_np - p_torch).max()
log(f"numpy vs torch 순전파 최대오차={max_diff:.2e} (1e-4 이하면 정상)")
assert max_diff < 1e-3, "numpy 이식 불일치"

common = dict(v66)
for k in list(common):
    if k.endswith("_weight") or k == "base_weight":
        common[k] = float(common[k]) * (1.0 - MLP_WEIGHT)
common["mlp_weights"] = mlp_weights
common["mlp_weight"] = MLP_WEIGHT
s = sum(float(v) for k, v in common.items() if k.endswith("_weight") or k == "base_weight")
assert abs(s - 1.0) < 1e-9, s
log(f"weights: mlp={MLP_WEIGHT:.3f} sum={s:.6f}")

out = os.path.join(OUT_DIR, "model_artifacts_v75.pkl")
joblib.dump(common, out)
log(f"저장: {out} ({os.path.getsize(out)/1e6:.1f}MB)")
log(f"총 {time.time()-t0:.0f}s")
