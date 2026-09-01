"""idea62 — idea61 MLP 실패 원인 분해.

idea61 결과: corr(mlp,v66)=0.807(직교성 통과) 이지만 단독 246 vs v47local 932.
진단: Cov/Var=0.583(심한 과분산), BCE 0.6916 vs baseline 0.6928(거의 무학습),
      최적축소해도 502까지밖에 회복 안 됨.

가설 3개를 ablation으로 분리한다:
  H1 loss 불일치 — 평가는 Brier인데 BCE로 학습했다. MSE로 직접 최적화하면?
  H2 임베딩 과적합 — valid unknown이 pitcher 20%/batter 9.6%. 임베딩을 빼면?
  H3 최적화 실패 — lr 2e-3에서 1 epoch만에 es가 바닥, 학습이 안 됐다. lr을 낮추고
     cosine schedule + 더 긴 patience로 제대로 태우면?

판정은 (a) 단독점수가 v47local의 90%(=839) 이상까지 오는가, (b) corr(v66)<0.90 유지.
둘 다 만족하는 변종만 프로덕션 후보. 로컬 blend 델타는 여전히 판정에 쓰지 않는다.
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
CD = "idea62_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
BATCH = 8192
MIN_COUNT = 30

CONFIGS = [
    dict(name="mse_emb8",     loss="mse", emb=8,  lr=5e-4, drop=0.15, ewd=1e-3, wd=1e-5, ep=40, pat=6),
    dict(name="mse_noemb",    loss="mse", emb=0,  lr=5e-4, drop=0.15, ewd=0.0,  wd=1e-5, ep=40, pat=6),
    dict(name="bce_emb8",     loss="bce", emb=8,  lr=5e-4, drop=0.15, ewd=1e-3, wd=1e-5, ep=40, pat=6),
    dict(name="mse_emb16_reg", loss="mse", emb=16, lr=5e-4, drop=0.30, ewd=5e-3, wd=1e-4, ep=40, pat=6),
]


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

UPTO, VAL = 2023, 2024
tr = season <= UPTO
va = season == VAL
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
v66 = (.1824 * base + .2432 * hur + .0608 * mr + .1216 * od + .1520 * mo
       + .08 * cb + .08 * cr + .08 * f5)
V47 = .30 * base + .40 * hur + .10 * mr + .20 * od
B47, B66 = sc(V47), sc(v66)
TARGET = B47 * 0.90
log(f"v47local={B47:.2f} v66local={B66:.2f}  통과선(v47의 90%)={TARGET:.2f}")

vc_p = pd.Series(pid[tr]).value_counts()
vc_b = pd.Series(bid[tr]).value_counts()
pmap = {v: i + 1 for i, v in enumerate(vc_p[vc_p >= MIN_COUNT].index)}
bmap = {v: i + 1 for i, v in enumerate(vc_b[vc_b >= MIN_COUNT].index)}
ip = np.array([pmap.get(v, 0) for v in pid], dtype=np.int64)
ib = np.array([bmap.get(v, 0) for v in bid], dtype=np.int64)

mu = Xn[tr].mean(0)
sd = Xn[tr].std(0)
sd[sd < 1e-8] = 1.0
Z = np.clip((Xn - mu) / sd, -10, 10)
tr_idx = np.where(tr)[0]
cut = int(len(tr_idx) * 0.92)
fit_i, es_i = tr_idx[:cut], tr_idx[cut:]
va_i = np.where(va)[0]
w_rec = (0.5 ** ((UPTO - season) / 2.0)).astype(np.float32)
T = lambda a, i: torch.as_tensor(a[i])


class Net(nn.Module):
    def __init__(self, n_num, n_p, n_b, emb, drop):
        super().__init__()
        self.use_emb = emb > 0
        d = n_num
        if self.use_emb:
            self.ep = nn.Embedding(n_p, emb)
            self.eb = nn.Embedding(n_b, emb)
            nn.init.normal_(self.ep.weight, 0, 0.01)
            nn.init.normal_(self.eb.weight, 0, 0.01)
            d += 2 * emb
        self.mlp = nn.Sequential(
            nn.Linear(d, 256), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(drop),
            nn.Linear(128, 1),
        )

    def forward(self, xn, xp, xb):
        h = torch.cat([xn, self.ep(xp), self.eb(xb)], 1) if self.use_emb else xn
        return self.mlp(h).squeeze(1)


def run(cfg):
    log(f"--- {cfg['name']}: loss={cfg['loss']} emb={cfg['emb']} lr={cfg['lr']} drop={cfg['drop']} ---")
    torch.manual_seed(42)
    net = Net(Xn.shape[1], len(pmap) + 1, len(bmap) + 1, cfg["emb"], cfg["drop"])
    groups = [{"params": list(net.mlp.parameters()), "weight_decay": cfg["wd"]}]
    if cfg["emb"] > 0:
        groups.append({"params": list(net.ep.parameters()) + list(net.eb.parameters()),
                       "weight_decay": cfg["ewd"]})
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["ep"])
    bce = nn.BCEWithLogitsLoss(reduction="none")

    def loss_fn(out, yt):
        return bce(out, yt) if cfg["loss"] == "bce" else (torch.sigmoid(out) - yt) ** 2

    ez, ep_, eb_ = T(Z, es_i), T(ip, es_i), T(ib, es_i)
    ey, ew = T(y, es_i), T(w_rec, es_i)
    best, best_state, bad = np.inf, None, 0
    for e in range(cfg["ep"]):
        net.train()
        perm = np.random.default_rng(42 + e).permutation(len(fit_i))
        for s in range(0, len(perm), BATCH):
            j = fit_i[perm[s:s + BATCH]]
            opt.zero_grad()
            w = T(w_rec, j)
            loss = (loss_fn(net(T(Z, j), T(ip, j), T(ib, j)), T(y, j)) * w).sum() / w.sum()
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            vl = float(((loss_fn(net(ez, ep_, eb_), ey) * ew).sum() / ew.sum()))
        if vl < best - 1e-7:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if e % 5 == 0 or bad >= cfg["pat"]:
            log(f"    epoch {e:2d} es={vl:.6f} best={best:.6f} bad={bad}")
        if bad >= cfg["pat"]:
            break
    net.load_state_dict(best_state)
    net.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(va_i), 65536):
            j = va_i[s:s + 65536]
            out.append(torch.sigmoid(net(T(Z, j), T(ip, j), T(ib, j))).numpy())
    p = np.concatenate(out).astype(np.float64)
    np.save(f"{CD}/A_{cfg['name']}.npy", p)

    Vp = np.mean((p - r) ** 2)
    Cov = np.mean((yv - r) * (p - r))
    k = Cov / Vp
    p_cal = r + k * (p - r)
    c66 = np.corrcoef(p, v66)[0, 1]
    log(f"  단독={sc(p):.2f}  캘리브후={sc(p_cal):.2f}  Cov/Var={k:.4f}  corr(v66)={c66:.4f}  sd={p.std():.5f}")
    return dict(name=cfg["name"], solo=sc(p), cal=sc(p_cal), k=k, c66=c66)


res = [run(c) for c in CONFIGS]
print()
print("=" * 86)
print(f"{'변종':16s} {'단독':>9s} {'캘리브후':>9s} {'Cov/Var':>9s} {'corr(v66)':>10s}  판정")
for x in res:
    ok = (x["cal"] >= TARGET) and (x["c66"] < 0.90)
    print(f"{x['name']:16s} {x['solo']:9.2f} {x['cal']:9.2f} {x['k']:9.4f} {x['c66']:10.4f}  "
          f"{'통과' if ok else '기각'}")
print(f"\n통과선: 캘리브후 >= {TARGET:.2f} (v47local의 90%) AND corr(v66) < 0.90")
print(f"참고: v47local={B47:.2f} v66local={B66:.2f}")
log(f"총 {time.time()-t0:.0f}s")
