"""idea51 — MTEN (Multi-Task Entity Network). 우리가 직접 설계한 신경망 구조.

## 왜 기존 NN은 실패했는가 (이력 분석)
  초기(약한 GBDT): HGB0.75+NN0.25 -> 로컬 +26.5
  v26(강한 GBDT):  최적 NN가중치 = 0.0, 강제혼합시 -33~-82
  시도된 것: MLP, embedding-MLP(phase3/19), TabM(phase67) — **전부 y 하나만 예측**.
  즉 GBDT와 '같은 목표를 같은 방식으로' 겨뤄서 졌다.

## 설계 원리 (이 프로젝트에서 실증된 것만 사용)
  1) multi-task aux head가 우리 문제의 유일한 작동 메커니즘
     (midother 실측 +2.37, multires +10.17, ordinal +5.08, unified5 +6.99)
     GBDT에선 '공유 트리구조'로 구현됐다. NN에선 '공유 은닉표현'으로 구현한다.
  2) LGB/XGB가 실패한 이유는 corr(base)=0.943 — 다양성이 없어서다.
     NN은 귀납편향이 근본적으로 달라 상관이 훨씬 낮을 것으로 기대.
     **단독성능이 낮아도 상관만 낮으면 블렌드에 기여한다.**
  3) 엔티티 임베딩: 트리는 pitcher_id를 타겟인코딩된 스칼라로만 본다.
     임베딩은 투수당 16차원 벡터를 학습 -> 트리가 표현 못 하는 축.

## 구조
  입력: 162피처(표준화) + pitcher_emb(16) + batter_emb(16)
  트렁크: 194 -> 256 -> 128  (BatchNorm + GELU + Dropout, residual connection)
  헤드 3개: y / 1-middle / 1-other   <- midother와 정확히 동일한 타겟
  손실: NaN 마스킹 MSE (MultiRMSEWithMissingValues와 동일 의미)
  출력: head0 sigmoid -> 확률

## 검증
  fold A(train<=2023 -> 2024), 규약대로 (1-w)*v47local + w*NN, 전체2024.
  핵심 지표는 점수보다 **corr(NN, base)** — 0.943보다 낮아야 의미 있음.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CD = "idea51_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
torch.set_num_threads(os.cpu_count() or 4)
SEEDS = [42, 7]
EMB = 16
EPOCHS = 12
BATCH = 4096


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("데이터 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float64)
seasons = meta["season"].to_numpy(np.float64)
pid_raw = meta["pitcher_id"].to_numpy()
bid_raw = meta["batter_id"].to_numpy()
n_ = meta["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
order = meta.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
same_next = np.zeros(len(meta), dtype=bool)
same_next[order[:-1]] = (pid_raw[order][1:] == pid_raw[order][:-1])


def recover(col):
    c = np.round(meta[col].fillna(0).to_numpy(np.float64) * n_)
    co = c[order]
    d = np.empty(len(meta)); d[:-1] = co[1:] - co[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(len(meta)); lab[order] = d
    return lab


lab_rev = recover("asof_pitcher_reverse_rate")
lab_mid = recover("asof_pitcher_middle_rate")
valid = ~(np.isnan(lab_rev) | np.isnan(lab_mid))
tot = y + lab_rev + lab_mid
lab_other = np.where(valid, (tot == 0).astype(np.float64), np.nan)
H = np.column_stack([y,
                     np.where(valid, 1.0 - lab_mid, np.nan),
                     np.where(valid, 1.0 - lab_other, np.nan)]).astype(np.float32)

pmap = {v: i for i, v in enumerate(np.unique(pid_raw))}
bmap = {v: i for i, v in enumerate(np.unique(bid_raw))}
pidx = np.array([pmap[v] for v in pid_raw], dtype=np.int64)
bidx = np.array([bmap[v] for v in bid_raw], dtype=np.int64)
log(f"  투수 {len(pmap)}명 / 타자 {len(bmap)}명 / 피처 {X.shape[1]}")

tr_m = seasons <= 2023
va_m = seasons == 2024
Xn = X.to_numpy(np.float32)
mu = Xn[tr_m].mean(0); sd = Xn[tr_m].std(0); sd[sd < 1e-6] = 1.0
Xn = np.nan_to_num((Xn - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
w_rec = (0.5 ** ((2023 - seasons) / 2.0)).astype(np.float32)

yv = y[va_m]; r = yv.mean(); BS = r * (1 - r)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / BS)
A = lambda ps: np.mean([np.load(p) for p in ps], axis=0)
base = A([f"phase90_cache/A_base_{n}.npy" for n in ["d6", "d8", "sub"]])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ["d6", "d8"]], axis=0)
mr = A([f"idea13_cache/A_multires_s{k}.npy" for k in [42, 7]])
od = A([f"idea13_cache/A_ordinal_s{k}.npy" for k in [42, 7]])
V47 = 0.30 * base + 0.40 * hur + 0.10 * mr + 0.20 * od
B47 = sc(V47)
uni = A([f"idea46_cache/A_midother_s{k}.npy" for k in [42, 7]])
log(f"  v47local={B47:.2f}  midother단독={sc(uni):.2f}  corr(midother,base)={np.corrcoef(uni,base)[0,1]:.4f}")


class MTEN(nn.Module):
    def __init__(self, nf, npit, nbat, emb=EMB, h1=256, h2=128, p=0.15):
        super().__init__()
        self.pe = nn.Embedding(npit, emb)
        self.be = nn.Embedding(nbat, emb)
        nn.init.normal_(self.pe.weight, 0, 0.01)
        nn.init.normal_(self.be.weight, 0, 0.01)
        d = nf + 2 * emb
        self.l1 = nn.Sequential(nn.Linear(d, h1), nn.BatchNorm1d(h1), nn.GELU(), nn.Dropout(p))
        self.l2 = nn.Sequential(nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.GELU(), nn.Dropout(p))
        self.skip = nn.Linear(d, h2)
        self.heads = nn.Linear(h2, 3)

    def forward(self, xc, pi, bi):
        z = torch.cat([xc, self.pe(pi), self.be(bi)], 1)
        h = self.l2(self.l1(z)) + self.skip(z)
        return self.heads(h)


def masked_mse(out, tgt, wt):
    m = torch.isfinite(tgt)
    d = torch.where(m, out - torch.nan_to_num(tgt), torch.zeros_like(out)) ** 2
    return (d * wt.unsqueeze(1)).sum() / (m.float() * wt.unsqueeze(1)).sum().clamp(min=1.0)


tr_idx = np.where(tr_m)[0]
va_idx = np.where(va_m)[0]
Xva_t = torch.from_numpy(Xn[va_idx])
pva_t = torch.from_numpy(pidx[va_idx]); bva_t = torch.from_numpy(bidx[va_idx])

preds = []
for seed in SEEDS:
    f = f"{CD}/A_mten_s{seed}.npy"
    if os.path.exists(f):
        preds.append(np.load(f)); log(f"  s{seed} 캐시"); continue
    ts = time.time()
    torch.manual_seed(seed); np.random.seed(seed)
    net = MTEN(Xn.shape[1], len(pmap), len(bmap))
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=2e-3, total_steps=EPOCHS * (len(tr_idx) // BATCH + 1))
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(tr_idx)
        tot_l, nb = 0.0, 0
        for i in range(0, len(perm), BATCH):
            b = perm[i:i + BATCH]
            if len(b) < 32:
                continue
            opt.zero_grad()
            out = net(torch.from_numpy(Xn[b]), torch.from_numpy(pidx[b]), torch.from_numpy(bidx[b]))
            loss = masked_mse(out, torch.from_numpy(H[b]), torch.from_numpy(w_rec[b]))
            loss.backward(); opt.step(); sch.step()
            tot_l += float(loss); nb += 1
        net.eval()
        with torch.no_grad():
            pv = torch.sigmoid(net(Xva_t, pva_t, bva_t)[:, 0] * 4 - 2).numpy().astype(np.float64)
        log(f"    [s{seed}] ep{ep+1:02d} loss={tot_l/max(nb,1):.5f} 검증={sc(pv):.1f}")
    with torch.no_grad():
        raw = net(Xva_t, pva_t, bva_t)[:, 0].numpy().astype(np.float64)
    p = np.clip(raw, 0.0, 1.0)
    np.save(f, p); preds.append(p)
    log(f"  s{seed} 완료 ({time.time()-ts:.0f}s) 단독={sc(p):.2f}")

nnp = np.mean(preds, axis=0)
print()
print("=" * 72)
print(f"MTEN 단독 = {sc(nnp):.2f}   (midother {sc(uni):.2f} / base {sc(base):.2f})")
print(f"corr(MTEN, base)     = {np.corrcoef(nnp, base)[0,1]:.4f}   <- 낮을수록 다양성 이득")
print(f"corr(MTEN, midother) = {np.corrcoef(nnp, uni)[0,1]:.4f}")
print(f"(참고) corr(lgb,base)=0.9430 / corr(midother,base)={np.corrcoef(uni,base)[0,1]:.4f}")
print("=" * 72)
V60A = 0.80 * V47 + 0.20 * uni
b60a = sc(V60A)
print(f"\nv60a 로컬 기준 = {b60a:.2f}")
for wv in [0.03, 0.05, 0.08, 0.10, 0.15]:
    v = sc((1 - wv) * V60A + wv * nnp)
    print(f"  +MTEN w={wv:.2f}  로컬={v:8.2f}  Δ={v-b60a:+.2f}")
log(f"총 {time.time()-t0:.0f}s")
