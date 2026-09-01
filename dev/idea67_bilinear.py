"""idea67 — pitcher x batter 명시적 bilinear 항 추가.

지금까지 확정된 사실들:
  - idea61/62: concat+MLP 구조로 corr(v66)=0.88 확보했지만 단독성능 부족(711)
  - idea63: 시드/배치 앙상블 8종 전부 mse_emb8(idea62) 단일시드보다 나쁨 -> 앙상블 기각
  - idea64: 임베딩을 GBDT피처로 얹으면 -145(과적합) -> 축소없는 raw 임베딩 위험 확인
  - idea66: corr penalty는 rho보다 정확도가 더 빨리 무너져 전 구간 손해 -> 기각

사용자 제안(4개) 중 가장 근거가 강한 것만 반영: concat MLP는 pitcher x batter
상호작용을 dense layer에 맡겨 간접적으로만 학습한다. 명시적 bilinear 항
dot(pitcher_emb2, batter_emb2)을 로짓에 직접 더하면 이 핵심 상호작용을 구조적으로
강제할 수 있다. count/hand 교차항은 넣지 않는다 -- 그건 이미 162피처의
game_context 블록(활용도 24.8%)이 잘 처리하는 영역이라 트리가 못 잡는 부분이 아니다.

logit = mlp(num_features, emb_p, emb_b) + scale * dot(bi_p, bi_b)
bi_p/bi_b는 emb_p/emb_b와 별도의 저랭크(4~8차원) 임베딩 -- 표현공간을 분리해
MLP쪽 표현과 얽히지 않게 한다.

기준(mse_emb8, 단독=711, 캘리브후=711 근방, rho=0.999052)과 정확히 같은 조건에서
비교한다.
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
CD = "idea67_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
MIN_COUNT = 30
EMB = 8       # concat용 (idea62와 동일)
BIRANK = 8    # bilinear 전용 저랭크


def log(m):
    print(f"[{time.time()-t0:5.0f}s] {m}", flush=True)


X = pd.read_parquet("featcache_X.parquet")
meta = pd.read_parquet("featcache_meta.parquet")
y = meta["control_success"].to_numpy(np.float32)
season = meta["season"].to_numpy(np.float64)
pid = meta["pitcher_id"].to_numpy()
bid = meta["batter_id"].to_numpy()
Xn = X.to_numpy(np.float32)

UPTO, VAL = 2023, 2024
tr, va = season <= UPTO, season == VAL
yv = y[va].astype(np.float64)
r = float(yv.mean())
bs = r * (1 - r)
sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / bs)

avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
base = avg([f"phase90_cache/A_base_{n}.npy" for n in ("d6", "d8", "sub")])
hur = np.mean([(1 - np.load(f"phase90_cache/A_core_{n}.npy")) *
               np.load(f"phase90_cache/A_snc_{n}.npy") for n in ("d6", "d8")], axis=0)
mr = avg([f"idea13_cache/A_multires_s{s}.npy" for s in (42, 7)])
od = avg([f"idea13_cache/A_ordinal_s{s}.npy" for s in (42, 7)])
mo = avg([f"idea46_cache/A_midother_s{s}.npy" for s in (42, 7)])
cb = avg([f"idea54_cache/A_cond_ball_s{s}.npy" for s in (42, 7)])
cr = avg([f"idea54_cache/A_count_resid_s{s}.npy" for s in (42, 7)])
f5 = avg([f"idea54_cache/A_future50_multi_s{s}.npy" for s in (42, 7)])
v66_va = (.1824 * base + .2432 * hur + .0608 * mr + .1216 * od + .1520 * mo
          + .08 * cb + .08 * cr + .08 * f5)
B66 = sc(v66_va)
e1 = v66_va - yv
B1 = float(np.mean(e1 ** 2))
log(f"v66local={B66:.2f}  B1={B1:.6f}")

vc_p = pd.Series(pid[tr]).value_counts()
vc_b = pd.Series(bid[tr]).value_counts()
pmap = {v: i + 1 for i, v in enumerate(vc_p[vc_p >= MIN_COUNT].index)}
bmap = {v: i + 1 for i, v in enumerate(vc_b[vc_b >= MIN_COUNT].index)}
ip = np.array([pmap.get(v, 0) for v in pid], dtype=np.int64)
ib = np.array([bmap.get(v, 0) for v in bid], dtype=np.int64)
mu, sd_ = Xn[tr].mean(0), Xn[tr].std(0)
sd_[sd_ < 1e-8] = 1.0
Z = np.clip((Xn - mu) / sd_, -10, 10)
tr_idx = np.where(tr)[0]
cut = int(len(tr_idx) * 0.92)
fit_i, es_i = tr_idx[:cut], tr_idx[cut:]
va_i = np.where(va)[0]
w_rec = (0.5 ** ((UPTO - season) / 2.0)).astype(np.float32)
T = lambda a, i: torch.as_tensor(a[i])


class NetConcat(nn.Module):
    """대조군: idea62 mse_emb8과 동일 (concat only)."""
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


class NetBilinear(nn.Module):
    """concat MLP + 별도 저랭크 bilinear(pitcher, batter) 항을 로짓에 직접 가산."""
    def __init__(self, n_num, n_p, n_b):
        super().__init__()
        self.ep = nn.Embedding(n_p, EMB)
        self.eb = nn.Embedding(n_b, EMB)
        nn.init.normal_(self.ep.weight, 0, 0.01)
        nn.init.normal_(self.eb.weight, 0, 0.01)
        self.bip = nn.Embedding(n_p, BIRANK)
        self.bib = nn.Embedding(n_b, BIRANK)
        nn.init.normal_(self.bip.weight, 0, 0.01)
        nn.init.normal_(self.bib.weight, 0, 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(n_num + 2 * EMB, 256), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(128, 1),
        )
        self.bi_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, xn, xp, xb):
        h = self.mlp(torch.cat([xn, self.ep(xp), self.eb(xb)], 1)).squeeze(1)
        bi = (self.bip(xp) * self.bib(xb)).sum(1)
        return h + self.bi_scale * bi


def train_eval(NetCls, tag, emb_wd=1e-3, lr=5e-4, max_ep=40, pat=6, seed=42):
    log(f"--- {tag} (seed={seed}) ---")
    torch.manual_seed(seed)
    net = NetCls(Xn.shape[1], len(pmap) + 1, len(bmap) + 1)
    emb_params = list(net.ep.parameters()) + list(net.eb.parameters())
    groups = [
        {"params": list(net.mlp.parameters()), "weight_decay": 1e-5},
    ]
    if hasattr(net, "bip"):
        emb_params += list(net.bip.parameters()) + list(net.bib.parameters())
        groups.append({"params": [net.bi_scale], "weight_decay": 0.0, "lr": lr})
    groups.append({"params": emb_params, "weight_decay": emb_wd})
    opt = torch.optim.AdamW(groups, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_ep)
    ez, ep_, eb_, ey, ew = T(Z, es_i), T(ip, es_i), T(ib, es_i), T(y, es_i), T(w_rec, es_i)
    best, best_state, bad = np.inf, None, 0
    ts = time.time()
    for e in range(max_ep):
        net.train()
        perm = np.random.default_rng(seed * 1000 + e).permutation(len(fit_i))
        for s in range(0, len(perm), 8192):
            j = fit_i[perm[s:s + 8192]]
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
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= pat:
                break
    net.load_state_dict(best_state)
    if hasattr(net, "bi_scale"):
        log(f"    학습된 bi_scale={float(net.bi_scale):.4f}")
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(va_i), 65536):
            j = va_i[s:s + 65536]
            out.append(torch.sigmoid(net(T(Z, j), T(ip, j), T(ib, j))).numpy())
    p = np.concatenate(out).astype(np.float64)
    np.save(f"{CD}/A_{tag}.npy", p)

    Vp = np.mean((p - r) ** 2)
    k = np.mean((yv - r) * (p - r)) / Vp
    pc = r + k * (p - r)
    B2 = float(np.mean((pc - yv) ** 2))
    rho = float(np.corrcoef(e1, pc - yv)[0, 1])
    thr = np.sqrt(B1 / B2)
    bestw, bestd = 0.0, 0.0
    for w in np.linspace(0, 0.4, 161):
        d = sc((1 - w) * v66_va + w * pc) - B66
        if d > bestd:
            bestw, bestd = w, d
    log(f"  [{tag}] 단독={sc(p):.2f} ({time.time()-ts:.0f}s) 캘리브후={sc(pc):.2f} k={k:.3f} "
        f"rho={rho:.6f} 임계={thr:.6f} 여유={thr-rho:+.6f} | 최적w={bestw:.3f} {bestd:+.2f}")
    return dict(tag=tag, solo=sc(p), cal=sc(pc), rho=rho, thr=thr, w=bestw, d=bestd)


results = []
results.append(dict(tag="concat_ctrl", solo=672.10, cal=688.58, rho=0.999065, thr=0.998815, w=0.0, d=0.0))
results.append(train_eval(NetBilinear, "bilinear_v2"))    # bi_scale 학습버그 수정 후 재실행

print()
print("=" * 100)
print(f"{'변종':16s} {'단독':>9s} {'캘리브후':>9s} {'rho':>10s} {'임계':>10s} {'여유':>10s} {'최적w':>7s} {'로컬델타':>9s}")
for x in results:
    print(f"{x['tag']:16s} {x['solo']:9.2f} {x['cal']:9.2f} {x['rho']:10.6f} {x['thr']:10.6f} "
          f"{x['thr']-x['rho']:+10.6f} {x['w']:7.3f} {x['d']:+9.2f}")
log(f"총 {time.time()-t0:.0f}s")
