"""아키텍처 비교: Embedding-MLP / TabM(BatchEnsemble) / FT-Transformer(Attention).

기존 NN 시도(phase3_embedding_mlp, phase3_tabm)가 GBDT에 진 이유 3개를 전부 고친다.

  [문제1] recency weighting 미적용
      v25에서 half_life=2 sample_weight가 로컬 +22.6 / 실측 +5.3(981.44, 현재 최고점)으로
      검증됐는데 NN 실험은 전부 균등 가중이었다.

  [문제2] 멀티태스크 미사용
      pitchlabels.recover_pitch_labels로 reverse/middle/ball/strike 4개 라벨이 100.000%
      정확히 복원되는데(v17에서 확인) 조건부 테이블 만드는 데만 썼다. NN의 보조 타깃으로
      쓰면 행당 gradient 신호가 5배. BSS 0.01짜리 저신호 문제에서 멀티태스크는 정석.

  [문제3] 심각한 under-training
      phase3_tabm_config는 max_epochs=40인데 실제 학습은 epochs=[1~3]에서 멈췄다.
      es_tail_fraction=0.2 홀드아웃이 대표성이 없어 조기 종료가 너무 빨리 걸린 것으로 보인다.
      여기서는 시간순 tail 8%(GBDT의 time_split_es와 동일)로 맞추고 patience를 늘린다.

기준선 (2023->2024 폴드, 전부 team_te=expanding 동일 피처 91개):
  GBDT v25 (recency weighted)  = 861.9   <- 이기거나 근접해야 할 목표
  GBDT v23 (균등 가중)          = 839.3
  TabM base (기존)              = 740.5   (calib_diff +0.0101, epochs=2)
  embMLP    (기존)              =  29~489 (calib_diff +0.020, epochs=3)

핵심 가설: NN이 진 것은 표현력 부족이 아니라 (a) 시즌 드리프트 편향 (b) under-train 때문이고,
둘 다 고치면 GBDT와 대등해진다. 그러면 진짜 목적인 '이질적 모델 블렌딩'이 성립한다
(현재 HGB+CatBoost는 둘 다 depth6 GBDT라 사실상 같은 모델 -> 시드만 바꾼 v21이
피처 추가만큼 점수를 움직였던 이유).

실행: python phase61_nn_arch.py [mlp] [tabm] [ft]   (기본 mlp tabm)
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

from crosses import add_crosses
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from metrics import evaluate
from phase2_common import build_fold, time_split_es
from pitchlabels import LABELS, recover_pitch_labels
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon

SEED = 42
TRAIN_MAX, VALID_SEASON = 2023, 2024
HALF_LIFE = 2.0
AUX_WEIGHT = 0.3
ES_FRAC = 0.08
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CACHE = "phase61_cache"

INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
EMB_CAT_COLS = ["pitcher_id", "batter_id", "cat_game_type_raw", "base_state_raw", "count_state_raw"]

t0 = time.time()


def log(msg):
    print(f"[{time.time()-t0:6.0f}s] {msg}", flush=True)


# ======================================================================
# 1) 피처 — v25와 100% 동일한 91개 (team_te=expanding)
# ======================================================================

log("데이터 로드...")
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
log("베이스 피처 블록 준비 완료")

# 보조 타깃 (문제2) — train 전체에서 복원. 마지막 투구는 NaN -> 손실에서 마스킹.
log("투구 라벨 복원 (보조 타깃 4개)...")
dlab = recover_pitch_labels(df)
for c in dlab.columns:
    v = dlab[c]
    log(f"  {c}: 복원율 {100*v.notna().mean():.2f}%  평균 {v.mean():.4f}")


def stack(i, base_frame):
    X = pd.concat([base_frame.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                   dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True),
                   dpt.loc[i].reset_index(drop=True)], axis=1).astype(np.float64)
    return pd.concat([X, add_crosses(X), dly.loc[i].reset_index(drop=True)], axis=1)


log(f"폴드 구성 (train<={TRAIN_MAX}, valid={VALID_SEASON}, team_te=expanding)...")
tr_i = df[df.season <= TRAIN_MAX].index
va_i = df[df.season == VALID_SEASON].index
fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED,
                  include_team_te=True, team_te_mode="expanding")
y_tr, y_va = fold["y_train"], fold["y_valid"]
X_tr = stack(tr_i, fold["X_train"])
X_va = stack(va_i, fold["X_valid"])
FEATS = list(X_tr.columns)
log(f"피처 {len(FEATS)}개  train={len(X_tr):,}  valid={len(X_va):,}")


def recency_weight(seasons, half_life=HALF_LIFE):
    return 0.5 ** ((seasons.max() - seasons) / half_life)


w_tr = recency_weight(df.loc[tr_i, "season"].to_numpy(np.float64))
w_tr = w_tr / w_tr.mean()   # 평균 1로 정규화 (LR 스케일 안정)
log(f"recency weight: min={w_tr.min():.4f} max={w_tr.max():.4f}")


# ======================================================================
# 2) GBDT 기준선 — 블렌딩 상대 (캐시)
# ======================================================================

os.makedirs(CACHE, exist_ok=True)
GBDT_PRED = f"{CACHE}/gbdt_v25_valid_pred.npy"

if os.path.exists(GBDT_PRED):
    p_gbdt = np.load(GBDT_PRED)
    log(f"GBDT 기준선 캐시 로드 -> score={max(0, evaluate(y_va, p_gbdt)['bss']*1e5):.1f}")
else:
    log("GBDT 기준선(v25 구성) 학습...")
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
    log(f"GBDT 기준선 score={max(0, evaluate(y_va, p_gbdt)['bss']*1e5):.1f}")


# ======================================================================
# 3) NN 전처리 — 수치 quantile 변환 + 카테고리/ID 임베딩
# ======================================================================

log("NN 전처리 (quantile transform)...")
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000,
                         subsample=300_000, random_state=SEED)
Xn_tr = qt.fit_transform(X_tr.to_numpy(np.float64)).astype(np.float32)
Xn_va = qt.transform(X_va.to_numpy(np.float64)).astype(np.float32)
Xn_tr = np.nan_to_num(Xn_tr, nan=0.0, posinf=0.0, neginf=0.0)
Xn_va = np.nan_to_num(Xn_va, nan=0.0, posinf=0.0, neginf=0.0)

# 임베딩용 원시 카테고리 (train 어휘로 고정, unseen -> 0)
raw = df.loc[:, ["pitcher_id", "batter_id", "game_type", "base_state", "balls_before", "strikes_before"]].copy()
raw["cat_game_type_raw"] = raw["game_type"].astype(str)
raw["base_state_raw"] = raw["base_state"].astype(str)
raw["count_state_raw"] = (raw["balls_before"] * 4 + raw["strikes_before"]).astype(str)

cat_codes_tr, cat_codes_va, cat_sizes = [], [], []
for c in EMB_CAT_COLS:
    vocab = {v: i + 1 for i, v in enumerate(pd.unique(raw.loc[tr_i, c]))}   # 0 = unseen/pad
    cat_codes_tr.append(raw.loc[tr_i, c].map(vocab).fillna(0).to_numpy(np.int64))
    cat_codes_va.append(raw.loc[va_i, c].map(vocab).fillna(0).to_numpy(np.int64))
    cat_sizes.append(len(vocab) + 1)
    unseen = (cat_codes_va[-1] == 0).mean()
    log(f"  {c}: vocab={len(vocab)}  valid unseen={100*unseen:.1f}%")
C_tr = np.stack(cat_codes_tr, 1)
C_va = np.stack(cat_codes_va, 1)

# 보조 타깃 (NaN -> 마스크)
A_tr = dlab.loc[tr_i].to_numpy(np.float32)
A_mask_tr = (~np.isnan(A_tr)).astype(np.float32)
A_tr = np.nan_to_num(A_tr, nan=0.0)

N_NUM, N_AUX = Xn_tr.shape[1], A_tr.shape[1]
log(f"수치 {N_NUM}개 / 카테고리 {len(EMB_CAT_COLS)}개 / 보조타깃 {N_AUX}개 / device={DEVICE}")


# ======================================================================
# 4) 모델 3종
# ======================================================================

class CatEmbed(nn.Module):
    """카테고리/ID 임베딩. 차원은 카디널리티에 따라 자동."""

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


class BatchEnsembleLinear(nn.Module):
    """TabM 핵심 — 가중치 W는 k개 멤버가 공유하고, rank-1 어댑터 r/s와 bias만 멤버별로 둔다.
    파라미터는 거의 안 늘면서 k개 독립 예측을 얻는다 (BatchEnsemble, Wen et al. 2020)."""

    def __init__(self, d_in, d_out, k):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_in, d_out))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        # r/s를 ±1 랜덤 부호로 초기화 = 멤버 다양성의 원천 (TabM 논문 권장)
        self.r = nn.Parameter(torch.randint(0, 2, (k, d_in)).float() * 2 - 1)
        self.s = nn.Parameter(torch.randint(0, 2, (k, d_out)).float() * 2 - 1)
        self.bias = nn.Parameter(torch.zeros(k, d_out))

    def forward(self, x):            # x: (B, k, d_in)
        return torch.einsum("bkd,do->bko", x * self.r, self.weight) * self.s + self.bias


class TabM(nn.Module):
    """효율적 MLP 앙상블. k개 멤버가 각자 예측하고 확률 평균으로 합친다."""

    def __init__(self, n_num, cat_sizes, n_aux, k=32, d=512, n_blocks=3, dropout=0.1):
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

    def forward(self, xn, xc):
        x = torch.cat([xn, self.emb(xc)], dim=1)
        x = x.unsqueeze(1).expand(-1, self.k, -1)          # (B, k, d_in)
        for blk, nrm in zip(self.blocks, self.norms):
            x = self.drop(torch.relu(nrm(blk(x))))
        return self.head_main(x).squeeze(-1), self.head_aux(x)   # (B,k), (B,k,n_aux)


class FTTransformer(nn.Module):
    """각 피처를 토큰으로 보고 피처 간 attention. CLS 토큰으로 읽어낸다."""

    def __init__(self, n_num, cat_sizes, n_aux, d=64, n_blocks=2, n_heads=4, dropout=0.1):
        super().__init__()
        self.n_num = n_num
        self.num_w = nn.Parameter(torch.randn(n_num, d) * 0.02)
        self.num_b = nn.Parameter(torch.zeros(n_num, d))
        self.cat_embs = nn.ModuleList([nn.Embedding(s, d) for s in cat_sizes])
        self.cls = nn.Parameter(torch.randn(1, 1, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=d, nhead=n_heads, dim_feedforward=d * 2,
                                           dropout=dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=n_blocks)
        self.norm = nn.LayerNorm(d)
        self.head_main = nn.Linear(d, 1)
        self.head_aux = nn.Linear(d, n_aux)

    def forward(self, xn, xc):
        B = xn.shape[0]
        tok = xn.unsqueeze(-1) * self.num_w + self.num_b                 # (B, n_num, d)
        cat = torch.stack([e(xc[:, i]) for i, e in enumerate(self.cat_embs)], dim=1)
        h = torch.cat([self.cls.expand(B, -1, -1), tok, cat], dim=1)
        h = self.norm(self.enc(h)[:, 0])
        return self.head_main(h).squeeze(-1), self.head_aux(h)


# ======================================================================
# 5) 학습 루프 — 멀티태스크 + recency weight + 시간순 ES
# ======================================================================

def train_model(name, model, epochs=30, batch=4096, lr=2e-3, wd=1e-4, patience=6,
                subsample=None, is_tabm=False):
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    ti, ei = time_split_es(len(Xn_tr), frac=ES_FRAC)   # 시간순 tail을 ES 홀드아웃으로 (GBDT와 동일)
    if subsample is not None and subsample < len(ti):
        ti = np.random.RandomState(SEED).choice(ti, subsample, replace=False)
        ti.sort()
        log(f"  {name}: 학습 행 {len(ti):,}개로 서브샘플 (CPU 시간 제약)")

    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def to_t(a, dt=torch.float32):
        return torch.as_tensor(a, dtype=dt)

    Xn_t, C_t = to_t(Xn_tr), to_t(C_tr, torch.int64)
    y_t, w_t = to_t(y_tr.astype(np.float32)), to_t(w_tr.astype(np.float32))
    A_t, M_t = to_t(A_tr), to_t(A_mask_tr)

    def predict(idx_arr, bs=16384):
        model.eval()
        out = []
        with torch.no_grad():
            for s in range(0, len(idx_arr), bs):
                b = idx_arr[s:s + bs]
                xn = Xn_t[b].to(DEVICE)
                xc = C_t[b].to(DEVICE)
                lg, _ = model(xn, xc)
                p = torch.sigmoid(lg)
                if is_tabm:
                    p = p.mean(dim=1)          # k개 멤버 확률 평균
                out.append(p.cpu().numpy())
        return np.concatenate(out)

    def predict_valid(bs=16384):
        model.eval()
        out = []
        Xv, Cv = to_t(Xn_va), to_t(C_va, torch.int64)
        with torch.no_grad():
            for s in range(0, len(Xv), bs):
                lg, _ = model(Xv[s:s + bs].to(DEVICE), Cv[s:s + bs].to(DEVICE))
                p = torch.sigmoid(lg)
                if is_tabm:
                    p = p.mean(dim=1)
                out.append(p.cpu().numpy())
        return np.concatenate(out)

    best_es, best_state, bad = np.inf, None, 0
    hist = []
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
            if is_tabm:
                # 멤버별로 같은 타깃에 손실 -> 각 멤버가 독립 예측기로 학습됨
                loss_main = (bce(lg, yy.unsqueeze(1).expand_as(lg)).mean(1) * ww).mean()
                aa_e = aa.unsqueeze(1).expand_as(lg_aux)
                mm_e = mm.unsqueeze(1).expand_as(lg_aux)
                la = bce(lg_aux, aa_e) * mm_e
                loss_aux = (la.sum(dim=(1, 2)) / mm_e.sum(dim=(1, 2)).clamp(min=1) * ww).mean()
            else:
                loss_main = (bce(lg, yy) * ww).mean()
                la = bce(lg_aux, aa) * mm
                loss_aux = (la.sum(1) / mm.sum(1).clamp(min=1) * ww).mean()

            loss = loss_main + AUX_WEIGHT * loss_aux
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()); nb += 1
        sched.step()

        p_es = predict(ei)
        es_brier = float(np.mean((p_es - y_tr[ei]) ** 2))
        hist.append(es_brier)
        mark = ""
        if es_brier < best_es - 1e-7:
            best_es, bad = es_brier, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        log(f"  {name} ep{ep:02d}  loss={tot/max(nb,1):.5f}  ES_brier={es_brier:.6f}{mark}")
        if bad >= patience:
            log(f"  {name}: early stop (patience {patience})")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    p_va_nn = predict_valid()
    sc = max(0, evaluate(y_va, p_va_nn)["bss"] * 1e5)
    calib = float(p_va_nn.mean() - y_va.mean())
    log(f"  {name} 최종: score={sc:.1f}  calib_diff={calib:+.4f}  pred_std={p_va_nn.std():.4f}  epochs={len(hist)}")
    return p_va_nn, sc, calib


# ======================================================================
# 6) 실행 & 블렌딩
# ======================================================================

want = [a for a in sys.argv[1:] if a in ("mlp", "tabm", "ft")] or ["mlp", "tabm"]
log(f"실행 대상: {want}")

results = {}
gbdt_score = max(0, evaluate(y_va, p_gbdt)["bss"] * 1e5)

if "mlp" in want:
    log("=== Embedding MLP ===")
    m = EmbMLP(N_NUM, cat_sizes, N_AUX)
    results["mlp"] = train_model("mlp", m, epochs=30, batch=4096, lr=2e-3)

if "tabm" in want:
    log("=== TabM (k=8 BatchEnsemble, CPU 축소판) ===")
    m = TabM(N_NUM, cat_sizes, N_AUX, k=8, d=256, n_blocks=2)
    results["tabm"] = train_model("tabm", m, epochs=30, batch=8192, lr=2e-3, is_tabm=True)

if "ft" in want:
    log("=== FT-Transformer (feature attention) ===")
    m = FTTransformer(N_NUM, cat_sizes, N_AUX, d=64, n_blocks=2, n_heads=4)
    sub = None if DEVICE.type == "cuda" else 400_000
    results["ft"] = train_model("ft", m, epochs=20, batch=1024, lr=1e-3, subsample=sub)

log("\n" + "=" * 72)
log("요약 (2023->2024 폴드)")
log("=" * 72)
log(f"  GBDT v25 (기준선)      {gbdt_score:8.1f}   calib={p_gbdt.mean()-y_va.mean():+.4f}  std={p_gbdt.std():.4f}")
for k, (p, sc, cal) in results.items():
    log(f"  {k:<22} {sc:8.1f}   calib={cal:+.4f}  std={p.std():.4f}")

log("\n블렌딩 (GBDT + NN):")
best = None
for k, (p, sc, cal) in results.items():
    for wnn in (0.2, 0.3, 0.4, 0.5, 0.6):
        pb = (1 - wnn) * p_gbdt + wnn * p
        sb = max(0, evaluate(y_va, pb)["bss"] * 1e5)
        tag = f"  GBDT {1-wnn:.1f} + {k} {wnn:.1f}"
        log(f"{tag:<34} {sb:8.1f}   (기준선 대비 {sb-gbdt_score:+.1f})")
        if best is None or sb > best[0]:
            best = (sb, k, wnn)
if len(results) >= 2:
    ks = list(results)
    pb = 0.5 * p_gbdt + 0.5 * np.mean([results[k][0] for k in ks], axis=0)
    sb = max(0, evaluate(y_va, pb)["bss"] * 1e5)
    log(f"  GBDT 0.5 + NN평균({'+'.join(ks)}) 0.5   {sb:8.1f}   (기준선 대비 {sb-gbdt_score:+.1f})")
    if sb > best[0]:
        best = (sb, "nn_mean", 0.5)

if best:
    log(f"\n최고 블렌드: {best[1]} w={best[2]} -> {best[0]:.1f}  (GBDT 단독 {gbdt_score:.1f})")

os.makedirs(CACHE, exist_ok=True)
for k, (p, sc, cal) in results.items():
    np.save(f"{CACHE}/nn_{k}_valid_pred.npy", p)
with open(f"{CACHE}/phase61_summary.json", "w", encoding="utf-8") as f:
    json.dump({"gbdt": gbdt_score,
               **{k: {"score": v[1], "calib": v[2]} for k, v in results.items()}}, f,
              ensure_ascii=False, indent=2)
log(f"저장 완료: {CACHE}/")
