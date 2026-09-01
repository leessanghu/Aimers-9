"""투수/타자 엔티티 임베딩 신경망 — 트리가 원천적으로 못 보는 '정체성'을 가져온다.

왜: 트리는 pitcher_id를 count 인코딩으로만 받고, 실제로 쓰는 건 asof_success_rate라는 1차원
    요약이다. 플래툰/이닝/볼카운트를 손으로 깎은 게 전부 "1차원으론 부족하다"는 시도였고 2승 6패.
    임베딩은 투수당 잠재벡터를 학습해 여러 맥락과의 상호작용을 '투수들 간 구조를 공유하며' 잡는다.
    축소강도 K를 손으로 고를 필요도 없다(weight decay가 대신함).

미등장 엔티티: 최근 5시즌 기준 그 시즌 투구의 14~22%가 '이전 시즌에 없던 투수'다.
    -> index 0을 unknown으로 예약하고, 학습 중 확률 p로 무작위 마스킹해서
       unknown 벡터가 '평균적 투수'를 뜻하도록 훈련한다(안 하면 신규 20%에서 무너짐).

배포: 추론이 임베딩 조회 + 행렬곱뿐이라 학습 후 가중치를 numpy로 export하면
    requirements.txt에 torch를 넣지 않고 순수 numpy로 forward pass 가능.
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
TRAIN_MAX, VALID_SEASON = 2023, 2024
EMB_DIM = 8
UNK_P = 0.15          # 학습 중 unknown 마스킹 확률 (실제 미등장률 14~22%에 맞춤)
EPOCHS = 12
BATCH = 8192


class EmbedNet(nn.Module):
    def __init__(self, n_pit, n_bat, n_num, dim=EMB_DIM):
        super().__init__()
        self.ep = nn.Embedding(n_pit + 1, dim)   # 0 = unknown
        self.eb = nn.Embedding(n_bat + 1, dim)
        nn.init.normal_(self.ep.weight, 0, 0.01)
        nn.init.normal_(self.eb.weight, 0, 0.01)
        self.bn = nn.BatchNorm1d(n_num)
        # 임베딩 + FM식 내적 1개 + 수치피처
        self.mlp = nn.Sequential(
            nn.Linear(n_num + 2 * dim + 1, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(64, 1))

    def forward(self, xp, xb, xn):
        p, b = self.ep(xp), self.eb(xb)
        fm = (p * b).sum(1, keepdim=True)        # FM식 pitcher-batter 상호작용
        return self.mlp(torch.cat([self.bn(xn), p, b, fm], 1)).squeeze(1)


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

    fold = build_fold(df, TRAIN_MAX, VALID_SEASON, extra_features=None, seed=SEED, include_team_te=True)
    ytr, yva = fold["y_train"], fold["y_valid"]
    tr, va = df[df.season <= TRAIN_MAX].index, df[df.season == VALID_SEASON].index

    def stack(bf, i):
        return pd.concat([bf.reset_index(drop=True), dins.loc[i, INS].reset_index(drop=True),
                          dplt.loc[i].reset_index(drop=True), dinn.loc[i].reset_index(drop=True)],
                         axis=1).astype(np.float64)

    Xtr, Xva = stack(fold["X_train"], tr), stack(fold["X_valid"], va)
    print(f"수치피처 {Xtr.shape[1]}개  train={len(Xtr):,}  valid={len(Xva):,}  ({time.time()-t0:.0f}s)", flush=True)

    # 엔티티 인덱싱: train fold에 등장한 것만 1..K, 나머지는 0(unknown)
    pit_tr = df.loc[tr, "pitcher_id"]
    bat_tr = df.loc[tr, "batter_id"]
    pmap = {v: i + 1 for i, v in enumerate(sorted(pit_tr.unique()))}
    bmap = {v: i + 1 for i, v in enumerate(sorted(bat_tr.unique()))}
    xp_tr = pit_tr.map(pmap).fillna(0).to_numpy(np.int64)
    xb_tr = bat_tr.map(bmap).fillna(0).to_numpy(np.int64)
    xp_va = df.loc[va, "pitcher_id"].map(pmap).fillna(0).to_numpy(np.int64)
    xb_va = df.loc[va, "batter_id"].map(bmap).fillna(0).to_numpy(np.int64)
    print(f"투수 {len(pmap)}명 / 타자 {len(bmap)}명 임베딩,  "
          f"valid 미등장 비율 투수={np.mean(xp_va==0):.3f} 타자={np.mean(xb_va==0):.3f}", flush=True)

    mu, sd = Xtr.to_numpy().mean(0), Xtr.to_numpy().std(0) + 1e-8
    Ztr = ((Xtr.to_numpy() - mu) / sd).astype(np.float32)
    Zva = ((Xva.to_numpy() - mu) / sd).astype(np.float32)

    # 시간순 early-stopping 분할
    cut = int(len(Ztr) * 0.92)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    net = EmbedNet(len(pmap), len(bmap), Ztr.shape[1])
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

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
        tot = 0.0
        for s in range(0, n, BATCH):
            b = perm[s:s + BATCH]
            xpb, xbb = tp[b].clone(), tb[b].clone()
            # unknown 마스킹: unknown 임베딩이 '평균적 엔티티'를 의미하도록
            xpb[torch.rand(len(b)) < UNK_P] = 0
            xbb[torch.rand(len(b)) < UNK_P] = 0
            opt.zero_grad()
            out = net(xpb, xbb, tz[b])
            l = lossf(out, ty[b])
            l.backward()
            opt.step()
            tot += l.item() * len(b)
        net.eval()
        with torch.no_grad():
            vl = lossf(net(ep_, eb_, ez_), ey_).item()
        print(f"  epoch {ep+1:2d}  train={tot/n:.5f}  es={vl:.5f}  ({time.time()-t0:.0f}s)", flush=True)
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 2:
                print("  early stop", flush=True)
                break
    net.load_state_dict(best_state)

    net.eval()
    with torch.no_grad():
        p_nn = torch.sigmoid(net(vp, vb, vz)).numpy().astype(np.float64)

    m = evaluate(yva, p_nn)
    print(f"\n{'='*66}\nNN 단독  BSS={m['bss']:.6f}  score={max(0,m['bss']*100000):.1f}", flush=True)
    print(f"  pred mean={p_nn.mean():.4f} (실제 {yva.mean():.4f})  SD={p_nn.std():.4f}", flush=True)

    # 미등장/등장 구간별
    unk = xp_va == 0
    for nm, msk in [("투수 등장", ~unk), ("투수 미등장", unk)]:
        if msk.sum() > 100:
            print(f"  {nm:10s} n={msk.sum():7,}  BSS={evaluate(yva[msk],p_nn[msk])['bss']:.6f}", flush=True)

    np.save("phase19_nn_pred_2024.npy", p_nn)
    print(f"\n예측 저장: phase19_nn_pred_2024.npy  (phase18 GBM 예측과 상관/앙상블 비교용)", flush=True)
    print(f"총 {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
