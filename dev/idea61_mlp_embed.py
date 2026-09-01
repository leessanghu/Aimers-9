"""idea61 — MLP + entity embedding: 귀납편향이 다른 앙상블 멤버.

동기(PCA 진단): v60a 5멤버만으로도 PC1이 94.6%. 원인은 모든 멤버가 같은 귀납편향
(GBDT on 동일 162 집계피처)이기 때문. aux축(target reformation)을 아무리 늘려도
상관 0.98이라 N_eff≈1이고, 실측으로도 v65->v66 +1.05까지 수확체감이 확인됐다.

이 실험은 타깃이 아니라 **모델 클래스**를 바꾼다:
  - pitcher_id / batter_id를 저차원 임베딩으로 직접 학습 (트리는 target encoding으로
    1차원 요약만 쓰므로 pitcher x batter 상호작용을 구조적으로 못 잡는다)
  - MLP는 smooth/global 함수를 적합 -> 축상 분할(axis-aligned split)과 오차구조가 다름

**판정 기준은 로컬 점수가 아니다.** v47/v60a/v65/v66 4점에서 로컬-실측 Spearman이
-1.00(완전역전)으로 무너졌으므로 로컬 델타는 신뢰하지 않는다. 이 실험의 판정은:
  1) corr(mlp, v66) < 0.90 인가  <- 진짜 새 축인지 (기존 aux축은 전부 0.98)
  2) 단독 성능이 파국적이지 않은가 (v47local의 ~90% 이상)
둘 다 통과해야 프로덕션으로 간다.

추론 이식성: torch로 학습하되 가중치를 numpy로 뽑아 script.py에서 순수 numpy
행렬곱으로 추론한다 -> submit/requirements.txt 변경 불필요(환경 리스크 0).
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

torch.set_num_threads(max(1, (os.cpu_count() or 4) - 2))
CD = "idea61_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()

EMB_P, EMB_B = 16, 16
HID = (256, 128)
MIN_COUNT = 30        # 이보다 적게 등장한 id는 unknown(idx 0)으로 접는다
BATCH = 8192
MAX_EPOCH = 30
PATIENCE = 3
LR = 2e-3
WD = 1e-5
EMB_WD = 1e-4         # 임베딩이 투수실력을 통째로 외우는 것을 억제


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


log("캐시 로드...")
X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float32)
season = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
bid = meta["batter_id"].to_numpy()
Xn = X.to_numpy(np.float32)
FEATS = list(X.columns)


class Net(nn.Module):
    def __init__(self, n_num, n_p, n_b):
        super().__init__()
        self.ep = nn.Embedding(n_p, EMB_P)
        self.eb = nn.Embedding(n_b, EMB_B)
        nn.init.normal_(self.ep.weight, 0, 0.01)
        nn.init.normal_(self.eb.weight, 0, 0.01)
        d = n_num + EMB_P + EMB_B
        self.mlp = nn.Sequential(
            nn.Linear(d, HID[0]), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HID[0], HID[1]), nn.ReLU(), nn.Dropout(0.10),
            nn.Linear(HID[1], 1),
        )

    def forward(self, xn, xp, xb):
        return self.mlp(torch.cat([xn, self.ep(xp), self.eb(xb)], 1)).squeeze(1)


def build_index(ids_tr, min_count):
    """train에서 min_count 이상 등장한 id만 고유 인덱스. 0 = unknown."""
    vc = pd.Series(ids_tr).value_counts()
    keep = vc[vc >= min_count].index.tolist()
    return {v: i + 1 for i, v in enumerate(keep)}


def run_fold(tag, upto, val):
    log(f"===== fold {tag}: train<={upto} -> valid={val} =====")
    tr = season <= upto
    va = season == val
    yv = y[va].astype(np.float64)
    r = yv.mean()
    bs = r * (1 - r)
    sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / bs)

    pmap = build_index(pid[tr], MIN_COUNT)
    bmap = build_index(bid[tr], MIN_COUNT)
    ip = np.array([pmap.get(v, 0) for v in pid], dtype=np.int64)
    ib = np.array([bmap.get(v, 0) for v in bid], dtype=np.int64)
    log(f"  pitcher vocab={len(pmap)+1} batter vocab={len(bmap)+1}  "
        f"valid unknown: p={np.mean(ip[va]==0)*100:.1f}% b={np.mean(ib[va]==0)*100:.1f}%")

    mu = Xn[tr].mean(0)
    sd = Xn[tr].std(0)
    sd[sd < 1e-8] = 1.0
    Z = (Xn - mu) / sd
    np.clip(Z, -10, 10, out=Z)

    # 마지막 8%를 early-stopping split으로(기존 프로덕션 관례와 동일)
    tr_idx = np.where(tr)[0]
    cut = int(len(tr_idx) * 0.92)
    fit_i, es_i = tr_idx[:cut], tr_idx[cut:]
    w_rec = (0.5 ** ((upto - season) / 2.0)).astype(np.float32)

    dev = "cpu"
    net = Net(Xn.shape[1], len(pmap) + 1, len(bmap) + 1).to(dev)
    emb_params = list(net.ep.parameters()) + list(net.eb.parameters())
    mlp_params = list(net.mlp.parameters())
    opt = torch.optim.AdamW([
        {"params": emb_params, "weight_decay": EMB_WD},
        {"params": mlp_params, "weight_decay": WD},
    ], lr=LR)
    lossf = nn.BCEWithLogitsLoss(reduction="none")

    T = lambda a, i, d=None: torch.as_tensor(a[i], dtype=d) if d else torch.as_tensor(a[i])
    es_z, es_p, es_b = T(Z, es_i), T(ip, es_i), T(ib, es_i)
    es_y, es_w = T(y, es_i, torch.float32), T(w_rec, es_i, torch.float32)

    best, best_state, bad = np.inf, None, 0
    for ep in range(MAX_EPOCH):
        net.train()
        perm = np.random.default_rng(42 + ep).permutation(len(fit_i))
        tot = 0.0
        for s in range(0, len(perm), BATCH):
            j = fit_i[perm[s:s + BATCH]]
            opt.zero_grad()
            out = net(T(Z, j), T(ip, j), T(ib, j))
            w = T(w_rec, j, torch.float32)
            loss = (lossf(out, T(y, j, torch.float32)) * w).sum() / w.sum()
            loss.backward()
            opt.step()
            tot += float(loss) * len(j)
        net.eval()
        with torch.no_grad():
            vl = float(((lossf(net(es_z, es_p, es_b), es_y) * es_w).sum() / es_w.sum()))
        log(f"    epoch {ep:2d} train={tot/len(perm):.5f} es={vl:.5f}")
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                log(f"    early stop @ epoch {ep}")
                break
    net.load_state_dict(best_state)

    net.eval()
    preds = []
    va_i = np.where(va)[0]
    with torch.no_grad():
        for s in range(0, len(va_i), 65536):
            j = va_i[s:s + 65536]
            preds.append(torch.sigmoid(net(T(Z, j), T(ip, j), T(ib, j))).numpy())
    p_mlp = np.concatenate(preds).astype(np.float64)
    np.save(f"{CD}/{tag}_mlp.npy", p_mlp)

    avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
    base = avg([f"phase90_cache/{tag}_base_{n}.npy" for n in ("d6", "d8", "sub")])
    hur = np.mean([(1 - np.load(f"phase90_cache/{tag}_core_{n}.npy")) *
                   np.load(f"phase90_cache/{tag}_snc_{n}.npy") for n in ("d6", "d8")], axis=0)
    mr = avg([f"idea13_cache/{tag}_multires_s{s}.npy" for s in (42, 7)])
    od = avg([f"idea13_cache/{tag}_ordinal_s{s}.npy" for s in (42, 7)])
    mo = avg([f"idea46_cache/{tag}_midother_s{s}.npy" for s in (42, 7)])
    cb = avg([f"idea54_cache/{tag}_cond_ball_s{s}.npy" for s in (42, 7)])
    cr = avg([f"idea54_cache/{tag}_count_resid_s{s}.npy" for s in (42, 7)])
    f5 = avg([f"idea54_cache/{tag}_future50_multi_s{s}.npy" for s in (42, 7)])
    v66 = (.1824 * base + .2432 * hur + .0608 * mr + .1216 * od + .1520 * mo
           + .08 * cb + .08 * cr + .08 * f5)
    v47 = .30 * base + .40 * hur + .10 * mr + .20 * od

    c_v66 = np.corrcoef(p_mlp, v66)[0, 1]
    ab = X.loc[va, "x_ability_here"].to_numpy(np.float64)
    ins = X.loc[va, "inseason_success_smooth"].to_numpy(np.float64)
    log(f"  단독={sc(p_mlp):.2f}  (v47local={sc(v47):.2f} v66local={sc(v66):.2f})")
    log(f"  corr(mlp,v66)={c_v66:.4f}  corr(mlp,base)={np.corrcoef(p_mlp,base)[0,1]:.4f}  "
        f"corr(mlp,ability)={np.corrcoef(p_mlp,ab)[0,1]:.4f}  corr(mlp,inseason)={np.corrcoef(p_mlp,ins)[0,1]:.4f}")
    log(f"  Var: mlp={p_mlp.var():.6f} v66={v66.var():.6f}")
    for w in (0.05, 0.10, 0.15, 0.20, 0.30):
        log(f"    v66 blend w={w:.2f} -> {sc((1-w)*v66 + w*p_mlp)-sc(v66):+.3f} (참고용, 판정 아님)")
    return dict(tag=tag, solo=sc(p_mlp), corr_v66=c_v66)


res = [run_fold("A", 2023, 2024)]
print()
print("=" * 74)
print("판정: corr(mlp,v66) < 0.90 이면 진짜 새 축 (기존 aux축은 전부 0.98)")
for r in res:
    verdict = "통과 - 새 축" if r["corr_v66"] < 0.90 else "기각 - 기존 축과 중복"
    print(f"  fold {r['tag']}: 단독={r['solo']:.2f}  corr(v66)={r['corr_v66']:.4f}  -> {verdict}")
log(f"총 {time.time()-t0:.0f}s")
