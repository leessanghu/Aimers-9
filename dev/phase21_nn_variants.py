"""NN 변형 3종 — 상관 0.74의 출처를 규명하고 블렌딩 이득을 최대화.

1차 NN(phase19)은 epoch1에서 조기종료 + 임베딩 과적합(등장 BSS 0.0044 < 미등장 0.0057).
그런데도 HGB와 상관 0.7405, 20% 블렌드에서 +13.4. 제대로 만들면 더 나올 여지가 크다.

핵심 미확인 질문: 상관 0.74가 '임베딩' 덕분인가, 그냥 '신경망이라는 함수형태' 덕분인가.
  -> noemb 변형이 비슷한 상관/이득을 내면 임베딩은 불필요하고,
     미등장 20% 리스크와 배포 복잡도가 통째로 사라진다.

개선: lr 2e-3 -> 5e-4, 임베딩만 weight decay 강화(1e-2), patience 3, epochs 25.
"""
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from inning_split import build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from metrics import evaluate
from phase2_common import build_fold
from platoon import build_platoon_table, transform_platoon, K_PLATOON

torch.set_num_threads(2)
SEED = 42
INS = ["inseason_success_smooth", "inseason_ball_smooth", "inseason_reverse_smooth",
       "inseason_n", "inseason_is_first_appearance"]
EPOCHS, BATCH, PATIENCE = 25, 8192, 3


class Net(nn.Module):
    def __init__(self, n_pit, n_bat, n_num, dim=8, use_emb=True):
        super().__init__()
        self.use_emb = use_emb
        extra = 0
        if use_emb:
            self.ep = nn.Embedding(n_pit + 1, dim)
            self.eb = nn.Embedding(n_bat + 1, dim)
            nn.init.normal_(self.ep.weight, 0, 0.01)
            nn.init.normal_(self.eb.weight, 0, 0.01)
            extra = 2 * dim + 1
        self.bn = nn.BatchNorm1d(n_num)
        self.mlp = nn.Sequential(
            nn.Linear(n_num + extra, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1))

    def forward(self, xp, xb, xn):
        z = self.bn(xn)
        if self.use_emb:
            p, b = self.ep(xp), self.eb(xb)
            z = torch.cat([z, p, b, (p * b).sum(1, keepdim=True)], 1)
        return self.mlp(z).squeeze(1)


def train_variant(name, use_emb, dim, unk_p, Ztr, xp_tr, xb_tr, ytr, Zva, xp_va, xb_va, yva,
                  n_pit, n_bat, lr=5e-4, emb_wd=1e-2, t0=0.0):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = Net(n_pit, n_bat, Ztr.shape[1], dim=dim, use_emb=use_emb)
    emb_params = [p for n_, p in net.named_parameters() if n_.startswith(("ep.", "eb."))]
    oth_params = [p for n_, p in net.named_parameters() if not n_.startswith(("ep.", "eb."))]
    groups = [{"params": oth_params, "weight_decay": 1e-5}]
    if emb_params:
        groups.append({"params": emb_params, "weight_decay": emb_wd})
    opt = torch.optim.AdamW(groups, lr=lr)
    lossf = nn.BCEWithLogitsLoss()

    cut = int(len(Ztr) * 0.92)
    tp, tb, tz, ty = (torch.from_numpy(a) for a in
                      (xp_tr[:cut], xb_tr[:cut], Ztr[:cut], ytr[:cut].astype(np.float32)))
    ep_, eb_, ez_, ey_ = (torch.from_numpy(a) for a in
                          (xp_tr[cut:], xb_tr[cut:], Ztr[cut:], ytr[cut:].astype(np.float32)))
    vp, vb, vz = (torch.from_numpy(a) for a in (xp_va, xb_va, Zva))

    n = len(tz)
    best, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        net.train()
        perm = torch.randperm(n)
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            xpb, xbb = tp[b].clone(), tb[b].clone()
            if use_emb and unk_p > 0:
                xpb[torch.rand(len(b)) < unk_p] = 0
                xbb[torch.rand(len(b)) < unk_p] = 0
            opt.zero_grad()
            l = lossf(net(xpb, xbb, tz[b]), ty[b])
            l.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(ep_, eb_, ez_), ey_).item()
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
        if ep % 3 == 0 or bad >= PATIENCE:
            print(f"    [{name}] epoch {ep+1:2d} es={vl:.5f} best={best:.5f} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= PATIENCE:
            break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        p = torch.sigmoid(net(vp, vb, vz)).numpy().astype(np.float64)
    return p


def main():
    t0 = time.time()
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
    dinn = transform_inning(df, build_inning_table(df), build_inning_offset(df), pp, sr, k=570.0)
    fold = build_fold(df, 2023, 2024, extra_features=None, seed=SEED, include_team_te=True)
    ytr, yva = fold["y_train"], fold["y_valid"]
    tr, va = df[df.season <= 2023].index, df[df.season == 2024].index

    def stack(bf, i):
        return pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                          dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True)],
                         axis=1).astype(np.float64)

    Xtr, Xva = stack(fold["X_train"], tr), stack(fold["X_valid"], va)
    pmap = {v: i + 1 for i, v in enumerate(sorted(df.loc[tr, "pitcher_id"].unique()))}
    bmap = {v: i + 1 for i, v in enumerate(sorted(df.loc[tr, "batter_id"].unique()))}
    xp_tr = df.loc[tr, "pitcher_id"].map(pmap).fillna(0).to_numpy(np.int64)
    xb_tr = df.loc[tr, "batter_id"].map(bmap).fillna(0).to_numpy(np.int64)
    xp_va = df.loc[va, "pitcher_id"].map(pmap).fillna(0).to_numpy(np.int64)
    xb_va = df.loc[va, "batter_id"].map(bmap).fillna(0).to_numpy(np.int64)
    mu, sd = Xtr.to_numpy().mean(0), Xtr.to_numpy().std(0) + 1e-8
    Ztr = ((Xtr.to_numpy() - mu) / sd).astype(np.float32)
    Zva = ((Xva.to_numpy() - mu) / sd).astype(np.float32)
    print(f"준비완료 {Xtr.shape[1]}피처 ({time.time()-t0:.0f}s)", flush=True)

    p_hgb = np.load("phase19b_hgb_pred_2024.npy")
    b_h = evaluate(yva, p_hgb)["bss"]
    print(f"HGB baseline BSS={b_h:.6f} ({b_h*1e5:.1f})\n" + "=" * 70, flush=True)

    variants = [
        ("emb8_tuned", True, 8, 0.15),
        ("emb4_tuned", True, 4, 0.20),
        ("noemb", False, 8, 0.0),
    ]
    preds = {"nn1_orig": np.load("phase19_nn_pred_2024.npy")}
    for nm, ue, dim, up in variants:
        print(f"\n--- {nm} ---", flush=True)
        preds[nm] = train_variant(nm, ue, dim, up, Ztr, xp_tr, xb_tr, ytr,
                                  Zva, xp_va, xb_va, yva, len(pmap), len(bmap), t0=t0)
        np.save(f"phase21_pred_{nm}.npy", preds[nm])

    print(f"\n{'='*70}\n변형별 결과\n{'='*70}", flush=True)
    for nm, p in preds.items():
        b = evaluate(yva, p)["bss"]
        r = np.corrcoef(p_hgb, p)[0, 1]
        best_w, best_d = 0.0, 0.0
        for w in np.arange(0, 0.55, 0.05):
            d = 1e5 * (evaluate(yva, (1 - w) * p_hgb + w * p)["bss"] - b_h)
            if d > best_d:
                best_w, best_d = w, d
        print(f"  {nm:12s} 단독BSS={b:.6f}({max(0,b*1e5):6.1f})  상관r={r:.4f}  "
              f"최적w={best_w:.2f}  블렌드델타={best_d:+7.1f}  실제예상={best_d*0.47:+6.1f}", flush=True)

    # 여러 NN 동시 블렌딩
    print(f"\n{'='*70}\nHGB + 여러 NN 동시 블렌딩\n{'='*70}", flush=True)
    nn_names = [n for n in preds if n != "nn1_orig"]
    p_nn_avg = np.mean([preds[n] for n in nn_names], axis=0)
    for w in [0.1, 0.15, 0.2, 0.25, 0.3]:
        d = 1e5 * (evaluate(yva, (1 - w) * p_hgb + w * p_nn_avg)["bss"] - b_h)
        print(f"  HGB{1-w:.2f} + NN평균{w:.2f}  델타={d:+7.1f}  실제예상={d*0.47:+6.1f}", flush=True)
    print(f"\n총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
