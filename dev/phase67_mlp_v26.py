"""v26의 132피처 전체로 train<=2023->valid=2024 폴드에서 GBDT 기준선을 새로 잡고
MLP를 블렌드 파트너로 붙여 phase63식(Brier 분해)으로 실제 이득을 확인한다.

주의: phase64/64b의 GBDT 기준선(gbdt_v25_valid_pred.npy)은 91피처 버전이라 그대로 못 쓴다.
132피처로 다시 학습해서 비교해야 공정하다.
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import QuantileTransformer
from catboost import CatBoostClassifier

from count_split import build_count_table, transform_count, K_COUNT
from crosses import add_crosses
from career_volatility import build_volatility_table, transform_volatility, K_VOL
from formfeat import build_role_table, transform_form, transform_role
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchlabels import recover_pitch_labels
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import build_trackman_profile, transform_trackman

SEED = 42
TRAIN_MAX, VALID_SEASON = 2023, 2024
HALF_LIFE = 2.0
ES_FRAC = 0.08
AUX_WEIGHT = 0.3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = "phase67_cache"
TM_CACHE = "phase64_trackman_profile.parquet"

t0 = time.time()


def log(m):
    print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)


# ======================================================================
log("데이터/피처 구성 (v26과 동일 132피처)...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
g = float(df["control_success"].mean())
sr = sorted(df["season"].unique().tolist())

se = build_season_end_table(df)
dins = transform_inseason(df, se, g, sr)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
pp = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
dplt = transform_platoon(df, build_platoon_table(df), pp, sr, k=K_PLATOON)
dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=K_INNING)
dcnt = transform_count(df, build_count_table(df), pp, sr, k=K_COUNT)
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
dly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0)
dvol = transform_volatility(df, build_volatility_table(se), sr, k=K_VOL)
role_tbl = build_role_table(df)
drole = transform_role(df, role_tbl, sr)
base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
dform = transform_form(df, drole, dins["inseason_success_smooth"].to_numpy(np.float64), base_middle)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
dtm = transform_trackman(df, prof, sr)
dlab = recover_pitch_labels(df)
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]


def stack(i, base_frame):
    X = pd.concat([base_frame.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    parts = [X, add_crosses(X), dly.loc[i].reset_index(drop=True), dcnt.loc[i].reset_index(drop=True),
             dvol.loc[i].reset_index(drop=True), drole.loc[i].reset_index(drop=True),
             dform.loc[i].reset_index(drop=True), dtm.loc[i].reset_index(drop=True)]
    return pd.concat(parts, axis=1)


tr_i = df[df.season <= TRAIN_MAX].index
va_i = df[df.season == VALID_SEASON].index
fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED,
                  include_team_te=True, team_te_mode="expanding")
y_tr, y_va = fold["y_train"], fold["y_valid"]
X_tr = stack(tr_i, fold["X_train"])
X_va = stack(va_i, fold["X_valid"])
log(f"피처 {X_tr.shape[1]}개  train={len(X_tr):,}  valid={len(X_va):,}")


def recency_weight(seasons, half_life=HALF_LIFE):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


w_tr = recency_weight(df.loc[tr_i, "season"].to_numpy(np.float64))
w_tr = w_tr / w_tr.mean()

os.makedirs(CACHE, exist_ok=True)
GBDT_PRED = f"{CACHE}/gbdt_v26_valid_pred.npy"

if os.path.exists(GBDT_PRED):
    p_gbdt = np.load(GBDT_PRED)
    log(f"GBDT(132피처) 캐시 로드 -> score={max(0, evaluate(y_va, p_gbdt)['bss']*1e5):.1f}")
else:
    log("GBDT(132피처) 기준선 학습...")
    ti, ei = time_split_es(len(X_tr))
    h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                       l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                       n_iter_no_change=20, random_state=SEED)
    cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                            verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
    h.fit(X_tr, y_tr, sample_weight=w_tr)
    cb.fit(X_tr.iloc[ti], y_tr[ti], sample_weight=w_tr[ti], eval_set=(X_tr.iloc[ei], y_tr[ei]))
    p_gbdt = 0.5 * h.predict_proba(X_va)[:, 1] + 0.5 * cb.predict_proba(X_va)[:, 1]
    np.save(GBDT_PRED, p_gbdt)
    log(f"GBDT(132피처) score={max(0, evaluate(y_va, p_gbdt)['bss']*1e5):.1f}")

gbdt_score = max(0, evaluate(y_va, p_gbdt)["bss"] * 1e5)

# ======================================================================
log("NN 전처리...")
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                         subsample=300_000, random_state=SEED)
Xn_tr = np.nan_to_num(qt.fit_transform(X_tr.to_numpy(np.float64)), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
Xn_va = np.nan_to_num(qt.transform(X_va.to_numpy(np.float64)), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

EMB_CAT_COLS = ["pitcher_id", "batter_id", "cat_game_type_raw", "base_state_raw", "count_state_raw"]
raw = df.loc[:, ["pitcher_id", "batter_id", "game_type", "base_state", "balls_before", "strikes_before"]].copy()
raw["cat_game_type_raw"] = raw["game_type"].astype(str)
raw["base_state_raw"] = raw["base_state"].astype(str)
raw["count_state_raw"] = (raw["balls_before"] * 4 + raw["strikes_before"]).astype(str)

cat_codes_tr, cat_codes_va, cat_sizes = [], [], []
for c in EMB_CAT_COLS:
    vocab = {v: i + 1 for i, v in enumerate(pd.unique(raw.loc[tr_i, c]))}
    cat_codes_tr.append(raw.loc[tr_i, c].map(vocab).fillna(0).to_numpy(np.int64))
    cat_codes_va.append(raw.loc[va_i, c].map(vocab).fillna(0).to_numpy(np.int64))
    cat_sizes.append(len(vocab) + 1)
C_tr, C_va = np.stack(cat_codes_tr, 1), np.stack(cat_codes_va, 1)

A_tr = dlab.loc[tr_i].to_numpy(np.float32)
A_mask_tr = (~np.isnan(A_tr)).astype(np.float32)
A_tr = np.nan_to_num(A_tr, nan=0.0)
N_NUM, N_AUX = Xn_tr.shape[1], A_tr.shape[1]
log(f"수치 {N_NUM} / 카테고리 {len(EMB_CAT_COLS)} / 보조타깃 {N_AUX} / device={DEVICE}")


class CatEmbed(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.dims = [min(32, max(4, int(round(1.6 * s ** 0.56)))) for s in sizes]
        self.embs = nn.ModuleList([nn.Embedding(s, d) for s, d in zip(sizes, self.dims)])
        self.out_dim = sum(self.dims)

    def forward(self, c):
        return torch.cat([e(c[:, i]) for i, e in enumerate(self.embs)], dim=1)


class EmbMLP(nn.Module):
    def __init__(self, n_num, cat_sizes, n_aux, d=512, n_blocks=3, dropout=0.1):
        super().__init__()
        self.emb = CatEmbed(cat_sizes)
        layers, d_in = [], n_num + self.emb.out_dim
        for _ in range(n_blocks):
            layers += [nn.Linear(d_in, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout)]
            d_in = d
        self.body = nn.Sequential(*layers)
        self.head_main = nn.Linear(d, 1)
        self.head_aux = nn.Linear(d, n_aux)

    def forward(self, xn, xc):
        h = self.body(torch.cat([xn, self.emb(xc)], dim=1))
        return self.head_main(h).squeeze(-1), self.head_aux(h)


def train_mlp(epochs=30, batch=4096, lr=2e-3, wd=1e-4, patience=6):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    ti, ei = time_split_es(len(Xn_tr), frac=ES_FRAC)
    model = EmbMLP(N_NUM, cat_sizes, N_AUX).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    Xn_t = torch.as_tensor(Xn_tr, dtype=torch.float32)
    C_t = torch.as_tensor(C_tr, dtype=torch.int64)
    y_t = torch.as_tensor(y_tr.astype(np.float32))
    w_t = torch.as_tensor(w_tr.astype(np.float32))
    A_t = torch.as_tensor(A_tr)
    M_t = torch.as_tensor(A_mask_tr)

    def predict(Xn_arr, C_arr, bs=16384):
        model.eval()
        Xv = torch.as_tensor(Xn_arr, dtype=torch.float32)
        Cv = torch.as_tensor(C_arr, dtype=torch.int64)
        out = []
        with torch.no_grad():
            for s in range(0, len(Xv), bs):
                lg, _ = model(Xv[s:s + bs].to(DEVICE), Cv[s:s + bs].to(DEVICE))
                out.append(torch.sigmoid(lg).cpu().numpy())
        return np.concatenate(out)

    best_es, best_state, bad = np.inf, None, 0
    for ep in range(1, epochs + 1):
        model.train()
        perm = np.random.RandomState(SEED + ep).permutation(ti)
        tot, nb = 0.0, 0
        for s in range(0, len(perm), batch):
            b = perm[s:s + batch]
            xn, xc = Xn_t[b].to(DEVICE), C_t[b].to(DEVICE)
            yy, ww = y_t[b].to(DEVICE), w_t[b].to(DEVICE)
            aa, mm = A_t[b].to(DEVICE), M_t[b].to(DEVICE)
            lg, lg_aux = model(xn, xc)
            loss_main = (bce(lg, yy) * ww).mean()
            la = bce(lg_aux, aa) * mm
            loss_aux = (la.sum(1) / mm.sum(1).clamp(min=1) * ww).mean()
            loss = loss_main + AUX_WEIGHT * loss_aux
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        sched.step()
        p_es = predict(Xn_tr[ei], C_tr[ei])
        es_brier = float(np.mean((p_es - y_tr[ei]) ** 2))
        mark = ""
        if es_brier < best_es - 1e-7:
            best_es, bad = es_brier, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        log(f"  ep{ep:02d}  loss={tot/max(nb,1):.5f}  ES_brier={es_brier:.6f}{mark}")
        if bad >= patience:
            log(f"  early stop (patience {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    p_va_nn = predict(Xn_va, C_va)
    sc = max(0, evaluate(y_va, p_va_nn)["bss"] * 1e5)
    calib = float(p_va_nn.mean() - y_va.mean())
    log(f"MLP 최종: score={sc:.1f}  calib_diff={calib:+.4f}  pred_std={p_va_nn.std():.4f}")
    return p_va_nn, sc, calib


log("=== Embedding MLP (132피처, v26과 동일 데이터) ===")
p_mlp, mlp_score, mlp_calib = train_mlp(epochs=30, batch=4096, lr=2e-3)

# ======================================================================
log("\n" + "=" * 72)
log("phase63식 분해 — 블렌드 실제 이득")
log("=" * 72)


def decompose(y, p):
    r = y.mean()
    bsref = r * (1 - r)
    bias = p.mean() - r
    rho = np.corrcoef(p, y)[0, 1]
    return {"score": max(0, 1e5 * (1 - np.mean((p - y) ** 2) / bsref)),
            "potential": 1e5 * rho ** 2, "bias": bias}


def pair_max(y, preds):
    A = np.column_stack([np.ones(len(y))] + list(preds))
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    return 1e5 * (1 - resid.var() / y.var())


d_gbdt = decompose(y_va, p_gbdt)
d_mlp = decompose(y_va, p_mlp)
mx = pair_max(y_va, [p_gbdt, p_mlp])
log(f"  GBDT(132피처)   score={d_gbdt['score']:.1f}  잠재력={d_gbdt['potential']:.1f}")
log(f"  MLP(132피처)    score={d_mlp['score']:.1f}  잠재력={d_mlp['potential']:.1f}")
log(f"  결합상한(GBDT+MLP 최적선형)  = {mx:.1f}   (GBDT 단독 잠재력 대비 {mx-d_gbdt['potential']:+.1f})")

log("\n블렌딩 가중치 탐색:")
best = (d_gbdt["score"], 0.0)
for wnn in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5):
    pb = (1 - wnn) * p_gbdt + wnn * p_mlp
    sb = max(0, evaluate(y_va, pb)["bss"] * 1e5)
    log(f"  GBDT {1-wnn:.1f} + MLP {wnn:.1f}   {sb:8.1f}   (기준선 대비 {sb-d_gbdt['score']:+.1f})")
    if sb > best[0]:
        best = (sb, wnn)
log(f"\n최적 블렌드: MLP 가중치={best[1]}  score={best[0]:.1f}  (GBDT 단독 대비 {best[0]-d_gbdt['score']:+.1f})")

np.save(f"{CACHE}/mlp_v26_valid_pred.npy", p_mlp)
with open(f"{CACHE}/phase67_summary.json", "w", encoding="utf-8") as f:
    json.dump({"gbdt_score": d_gbdt["score"], "mlp_score": d_mlp["score"],
               "blend_ceiling": mx, "best_blend_w": best[1], "best_blend_score": best[0]},
              f, ensure_ascii=False, indent=2)
log(f"저장 완료: {CACHE}/")
