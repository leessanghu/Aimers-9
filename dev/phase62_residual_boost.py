"""Residual boosting: GBDT(v25) logit을 고정 offset으로 주고 NN이 그 위의 잔차만 학습.

블렌딩(phase61)과 다른 점 — 블렌딩은 두 모델이 서로 독립으로 전체를 예측해서 평균내지만,
여기서는 GBDT가 이미 설명한 부분(예: x_ability_here/season/cat_game_type 같은 강한 feature)은
NN이 자동으로 무시하고, GBDT가 못 살린 잔여 신호(magnitude 0.001대로 죽어있던 feature들)에만
NN의 학습 용량이 집중되도록 강제한다.

  최종 logit = GBDT_logit(offset, 고정) + NN(X)          <- NN은 offset만큼 뺀 나머지를 학습
  loss = BCE(sigmoid(최종 logit), y)

GBDT_logit을 학습 데이터에도 넣어야 하는데, in-sample(자기 자신 학습에 쓰인) 예측을 그대로
쓰면 GBDT가 이미 외운 노이즈까지 offset에 섞여 NN이 헛것을 잔차로 착각한다. 그래서 train
구간은 5-fold 시간순 OOF로 GBDT를 다시 학습해 정직한 offset을 만든다(valid는 원래 v25처럼
전체 train으로 학습한 모델 그대로 사용 — 이미 아무 정보도 안 새는 구간).
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
from catboost import CatBoostClassifier
from sklearn.preprocessing import QuantileTransformer

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchlabels import recover_pitch_labels
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
TRAIN_MAX, VALID_SEASON = 2023, 2024
HALF_LIFE = 2.0
ES_FRAC = 0.08
N_OOF_FOLDS = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = "phase61_cache"   # phase61과 캐시 공유 (같은 X_tr/X_va/GBDT 기준선)

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)


# ======================================================================
# 1) phase61과 동일한 피처/데이터 재구성
# ======================================================================

log("데이터/피처 재구성 (phase61과 동일)...")
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
dpt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), pp, g, sr)
dly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0)
dlab = recover_pitch_labels(df)
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]


def stack(i, base_frame):
    X = pd.concat([base_frame.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


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


w_tr_full = recency_weight(df.loc[tr_i, "season"].to_numpy(np.float64))
w_tr_full = w_tr_full / w_tr_full.mean()


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


# ======================================================================
# 2) 정직한 GBDT offset — train은 5-fold 시간순 OOF, valid는 캐시된 v25 예측 재사용
# ======================================================================

OOF_PATH = f"{CACHE}/gbdt_oof_logit_train.npy"
VALID_PRED_PATH = f"{CACHE}/gbdt_v25_valid_pred.npy"

if not os.path.exists(VALID_PRED_PATH):
    raise SystemExit("phase61_nn_arch.py를 먼저 실행해 GBDT 기준선 캐시를 만들어야 함")
p_gbdt_valid = np.load(VALID_PRED_PATH)
offset_valid_logit = logit(p_gbdt_valid)

if os.path.exists(OOF_PATH):
    offset_train_logit = np.load(OOF_PATH)
    log(f"GBDT train OOF offset 캐시 로드 (RMSE 체크: mean={offset_train_logit.mean():.4f})")
else:
    log(f"GBDT train OOF offset 생성 ({N_OOF_FOLDS}-fold 시간순)...")
    n = len(X_tr)
    fold_id = (np.arange(n) * N_OOF_FOLDS) // n   # 시간순 블록 분할 (셔플 안 함 — 미래->과거 누수 방지)
    offset_train_logit = np.zeros(n, dtype=np.float64)
    for k in range(N_OOF_FOLDS):
        oof_mask = fold_id == k
        fit_mask = ~oof_mask
        # 자기 자신을 뺀 나머지로 학습해서 oof_mask 구간을 예측 (train 내부 self-OOF)
        Xf, yf, wf = X_tr.iloc[fit_mask], y_tr[fit_mask], w_tr_full[fit_mask]
        ti_local = np.arange(len(Xf))
        cut = int(len(ti_local) * (1 - ES_FRAC))
        h = HistGradientBoostingClassifier(max_depth=6, max_leaf_nodes=31, max_iter=500, learning_rate=0.03,
                                           l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
                                           n_iter_no_change=20, random_state=SEED)
        cb = CatBoostClassifier(iterations=3000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, random_seed=SEED,
                                verbose=0, early_stopping_rounds=50, min_data_in_leaf=200, loss_function="Logloss")
        h.fit(Xf, yf, sample_weight=wf)
        cb.fit(Xf.iloc[:cut], yf[:cut], sample_weight=wf[:cut], eval_set=(Xf.iloc[cut:], yf[cut:]))
        p_oof = 0.5 * h.predict_proba(X_tr.iloc[oof_mask])[:, 1] + 0.5 * cb.predict_proba(X_tr.iloc[oof_mask])[:, 1]
        offset_train_logit[oof_mask] = logit(p_oof)
        log(f"  fold {k+1}/{N_OOF_FOLDS} 완료 (oof n={oof_mask.sum():,}, fit n={fit_mask.sum():,})")
    np.save(OOF_PATH, offset_train_logit)
    p_oof_all = 1 / (1 + np.exp(-offset_train_logit))
    log(f"OOF score(전체, 참고용) = {max(0, evaluate(y_tr, p_oof_all)['bss']*1e5):.1f}")

log(f"GBDT offset 준비 완료. train std={offset_train_logit.std():.4f}  valid std={offset_valid_logit.std():.4f}")


# ======================================================================
# 3) NN 전처리 (phase61과 동일 방식, 여기서 재계산 — 프로세스 독립 실행 대비)
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


# ======================================================================
# 4) 잔차 보정용 소형 TabM (offset이 이미 대부분 설명하므로 크게 안 키움)
# ======================================================================

class CatEmbed(nn.Module):
    def __init__(self, sizes):
        super().__init__()
        self.dims = [min(32, max(4, int(round(1.6 * s ** 0.56)))) for s in sizes]
        self.embs = nn.ModuleList([nn.Embedding(s, d) for s, d in zip(sizes, self.dims)])
        self.out_dim = sum(self.dims)

    def forward(self, c):
        return torch.cat([e(c[:, i]) for i, e in enumerate(self.embs)], dim=1)


class BatchEnsembleLinear(nn.Module):
    def __init__(self, d_in, d_out, k):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.r = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
        self.s = nn.Parameter(torch.randint(0, 2, (k, d_out)).float() * 2 - 1)
        self.bias = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):
        return torch.einsum("bkd,do->bko", x * self.r, self.weight) * self.s + self.bias


class ResidualTabM(nn.Module):
    """offset(GBDT logit, 고정 상수 취급)을 최종 logit에 그대로 더하고, 이 네트워크는
    보정치만 출력하도록 학습된다. 초기 출력을 0 근처로 시작해 GBDT를 그대로 재현하는
    지점에서 학습을 시작한다(= '망가뜨리지 않는' boosting 초기화)."""

    def __init__(self, n_num, cat_sizes, n_aux, k=16, d=256, n_blocks=2, dropout=0.1):
        super().__init__()
        self.k = k
        self.emb = CatEmbed(cat_sizes)
        d_in = n_num + self.emb.out_dim
        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(BatchEnsembleLinear(d_in, d, k))
            self.norms.append(nn.LayerNorm(d))
            d_in = d
        self.drop = nn.Dropout(dropout)
        self.head_main = BatchEnsembleLinear(d, 1, k)
        self.head_aux = BatchEnsembleLinear(d, n_aux, k)
        # 마지막 층을 0 근처로 초기화 -> 학습 시작 시 delta ~= 0, 최종 logit ~= offset(GBDT 그대로)
        nn.init.zeros_(self.head_main.weight)
        nn.init.zeros_(self.head_main.bias)

    def forward(self, xn, xc):
        x = torch.cat([xn, self.emb(xc)], dim=1)
        x = x.unsqueeze(1).expand(-1, self.k, -1)
        for blk, nrm in zip(self.blocks, self.norms):
            x = self.drop(torch.relu(nrm(blk(x))))
        return self.head_main(x).squeeze(-1), self.head_aux(x)


# ======================================================================
# 5) 학습 — delta만 학습, offset은 detach된 상수로 더함
# ======================================================================

AUX_WEIGHT = 0.3
EPOCHS, BATCH, LR, WD, PATIENCE = 30, 2048, 1.5e-3, 1e-4, 6

torch.manual_seed(SEED)
np.random.seed(SEED)

ti, ei = time_split_es(len(Xn_tr), frac=ES_FRAC)
model = ResidualTabM(N_NUM, cat_sizes, N_AUX, k=16, d=256, n_blocks=2).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
bce = nn.BCEWithLogitsLoss(reduction="none")

Xn_t = torch.as_tensor(Xn_tr, dtype=torch.float32)
C_t = torch.as_tensor(C_tr, dtype=torch.int64)
y_t = torch.as_tensor(y_tr.astype(np.float32))
w_t = torch.as_tensor(w_tr_full.astype(np.float32))
off_t = torch.as_tensor(offset_train_logit.astype(np.float32))
A_t = torch.as_tensor(A_tr)
M_t = torch.as_tensor(A_mask_tr)


def predict_delta(Xn_arr, C_arr, bs=16384):
    model.eval()
    Xv = torch.as_tensor(Xn_arr, dtype=torch.float32)
    Cv = torch.as_tensor(C_arr, dtype=torch.int64)
    out = []
    with torch.no_grad():
        for s in range(0, len(Xv), bs):
            lg, _ = model(Xv[s:s + bs].to(DEVICE), Cv[s:s + bs].to(DEVICE))
            out.append(lg.mean(dim=1).cpu().numpy())   # k멤버 평균 delta
    return np.concatenate(out)


best_es, best_state, bad = np.inf, None, 0
for ep in range(1, EPOCHS + 1):
    model.train()
    perm = np.random.RandomState(SEED + ep).permutation(ti)
    tot, nb = 0.0, 0
    for s in range(0, len(perm), BATCH):
        b = perm[s:s + BATCH]
        xn, xc = Xn_t[b].to(DEVICE), C_t[b].to(DEVICE)
        yy, ww, oo = y_t[b].to(DEVICE), w_t[b].to(DEVICE), off_t[b].to(DEVICE)
        aa, mm = A_t[b].to(DEVICE), M_t[b].to(DEVICE)

        delta, delta_aux = model(xn, xc)                       # (B,k), (B,k,n_aux)
        final_logit = oo.unsqueeze(1) + delta                  # offset은 상수로 취급(그래디언트 안 흐름, GBDT 파라미터 없음)
        loss_main = (bce(final_logit, yy.unsqueeze(1).expand_as(final_logit)).mean(1) * ww).mean()
        aa_e, mm_e = aa.unsqueeze(1).expand_as(delta_aux), mm.unsqueeze(1).expand_as(delta_aux)
        loss_aux = (((bce(delta_aux, aa_e) * mm_e).sum(dim=(1, 2))) / mm_e.sum(dim=(1, 2)).clamp(min=1) * ww).mean()
        loss = loss_main + AUX_WEIGHT * loss_aux

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        tot += float(loss.item()); nb += 1
    sched.step()

    delta_es = predict_delta(Xn_tr[ei], C_tr[ei])
    p_es = 1 / (1 + np.exp(-(offset_train_logit[ei] + delta_es)))
    es_brier = float(np.mean((p_es - y_tr[ei]) ** 2))
    mark = ""
    if es_brier < best_es - 1e-7:
        best_es, bad = es_brier, 0
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        mark = " *"
    else:
        bad += 1
    log(f"ep{ep:02d}  loss={tot/max(nb,1):.5f}  ES_brier={es_brier:.6f}  delta_std={delta_es.std():.4f}{mark}")
    if bad >= PATIENCE:
        log(f"early stop (patience {PATIENCE})")
        break

if best_state is not None:
    model.load_state_dict(best_state)

delta_valid = predict_delta(Xn_va, C_va)
p_final = 1 / (1 + np.exp(-(offset_valid_logit + delta_valid)))
score_final = max(0, evaluate(y_va, p_final)["bss"] * 1e5)
score_gbdt_only = max(0, evaluate(y_va, p_gbdt_valid)["bss"] * 1e5)

log("\n" + "=" * 72)
log("결과 (2023->2024 폴드)")
log("=" * 72)
log(f"  GBDT v25 단독            {score_gbdt_only:8.1f}")
log(f"  GBDT + residual-NN       {score_final:8.1f}   (delta {score_final-score_gbdt_only:+.1f})")
log(f"  delta(logit 보정) 통계: mean={delta_valid.mean():+.4f}  std={delta_valid.std():.4f}")

os.makedirs(CACHE, exist_ok=True)
np.save(f"{CACHE}/residual_boost_valid_pred.npy", p_final)
with open(f"{CACHE}/phase62_summary.json", "w", encoding="utf-8") as f:
    json.dump({"gbdt_only": score_gbdt_only, "gbdt_plus_residual_nn": score_final,
               "delta_mean": float(delta_valid.mean()), "delta_std": float(delta_valid.std())},
              f, ensure_ascii=False, indent=2)
log(f"저장 완료: {CACHE}/residual_boost_valid_pred.npy")
