"""NN v2: 과거 실패(idea61-65) 사인을 정면 수정한 tabular MLP. 밤샘 원샷 체인.

단계:
 1) fold A(≤2023 -> 2024) 2시드 학습 + 시드건강검사
 2) 표준 스크리닝: v126 기준, 6축(mc6/strk/xu/xr/lty/mc6aux) 직교화 + 순열대조군 z
 3) z>2 통과시: 전체데이터 프로덕션 2시드 재학습 + numpy weights export
    (추론은 torch 없이 numpy forward - GELU tanh근사로 train=inference 일치)

설계:
 - 아키텍처: [162피처 z클립 + 투수emb8 + 타자emb8] -> 512 LN GELU drop
             -> residual block(512) x2 -> head_y(1) + head_mc6(6)
 - 손실: w*BCE(y) + 0.5*w*CE(mc6, 유효행만), w=recency 0.5^(dt/2)
 - AdamW: 임베딩 wd=3e-2(통암기 방지), 나머지 wd=1e-4
 - lr: 1에폭 워밍업 -> 1e-3 -> cosine -> 1e-5, grad clip 1.0, batch 16384
 - 시드건강검사: eval Brier > 0.256(수축상수 수준)이면 그 시드 폐기
"""
import sys, time, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import torch
import torch.nn as nn

torch.set_num_threads(4)
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
DEV = 'cpu'

# ---------- 데이터 ----------
X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y_all = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95a['feature_order'])
X_raw = X_df[FEAT].astype(np.float64).to_numpy()
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['row_id', 'pitcher_id', 'batter_id', 'asof_pitcher_n', 'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid_raw = df['pitcher_id'].to_numpy()
bid_raw = df['batter_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
order = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[order[:-1]] = (pid_raw[order][1:] == pid_raw[order][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(n); lab[order] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
cls = np.full(n, -1, dtype=np.int64)
cls[valid & (mid > 0.5)] = 0
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y_all == 0)] = 2
cls[nd & (y_all == 1) & (ball > 0.5)] = 3
cls[nd & (y_all == 1) & (strike > 0.5)] = 4
cls[nd & (y_all == 1) & (inplay > 0.5)] = 5
log('데이터/라벨 준비 완료')


# ---------- 모델 ----------
class Net(nn.Module):
    def __init__(self, n_feat, n_p, n_b, emb=8, hid=512, drop=0.15):
        super().__init__()
        self.emb_p = nn.Embedding(n_p + 1, emb)
        self.emb_b = nn.Embedding(n_b + 1, emb)
        self.emb_drop = nn.Dropout(0.10)
        act = lambda: nn.GELU(approximate='tanh')
        self.inp = nn.Sequential(nn.Linear(n_feat + 2 * emb, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.b1 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.b2 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.head_y = nn.Linear(hid, 1)
        self.head_c = nn.Linear(hid, 6)

    def forward(self, xz, ip, ib):
        e = torch.cat([self.emb_drop(self.emb_p(ip)), self.emb_drop(self.emb_b(ib))], dim=1)
        h = self.inp(torch.cat([xz, e], dim=1))
        h = h + self.b1(h)
        h = h + self.b2(h)
        return self.head_y(h).squeeze(1), self.head_c(h)


def train_one(tr_mask, upto, seed, tag):
    """tr_mask 내 마지막 8%를 얼리스탑 평가로 사용. 학습된 모델과 예측함수 반환."""
    torch.manual_seed(seed); np.random.seed(seed)
    idx_tr_all = np.where(tr_mask)[0]
    n_es = int(len(idx_tr_all) * 0.92)
    idx_fit, idx_ev = idx_tr_all[:n_es], idx_tr_all[n_es:]

    mu = np.nanmean(X_raw[idx_fit], axis=0)
    sd = np.nanstd(X_raw[idx_fit], axis=0) + 1e-9
    def zx(rows):
        z = (X_raw[rows] - mu) / sd
        z = np.clip(np.nan_to_num(z, nan=0.0), -10, 10)
        return z.astype(np.float32)

    pmap = {v: i + 1 for i, v in enumerate(pd.unique(pid_raw[idx_fit]))}
    bmap = {v: i + 1 for i, v in enumerate(pd.unique(bid_raw[idx_fit]))}
    ip_all = np.array([pmap.get(v, 0) for v in pid_raw], dtype=np.int64)
    ib_all = np.array([bmap.get(v, 0) for v in bid_raw], dtype=np.int64)

    w_all = (0.5 ** ((upto - season) / 2.0)).astype(np.float32)
    yv32 = y_all.astype(np.float32)
    cls_t = cls.copy()

    net = Net(len(FEAT), len(pmap), len(bmap)).to(DEV)
    emb_params = list(net.emb_p.parameters()) + list(net.emb_b.parameters())
    other = [p for nmn, p in net.named_parameters() if not nmn.startswith('emb_')]
    opt = torch.optim.AdamW([
        dict(params=emb_params, weight_decay=3e-2),
        dict(params=other, weight_decay=1e-4)], lr=1e-3)

    EPOCHS, BATCH = 18, 16384
    steps_per_ep = math.ceil(len(idx_fit) / BATCH)
    total_steps = EPOCHS * steps_per_ep
    warm = steps_per_ep  # 1에폭 워밍업
    def lr_at(step):
        if step < warm:
            return 1e-3 * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * prog))

    bce = nn.BCEWithLogitsLoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)

    Xev = torch.from_numpy(zx(idx_ev))
    ipev = torch.from_numpy(ip_all[idx_ev]); ibev = torch.from_numpy(ib_all[idx_ev])
    yev = y_all[idx_ev]

    best = (1e9, None, -1)
    step = 0
    patience, bad = 4, 0
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(idx_fit)
        ep_loss, nb = 0.0, 0
        for s in range(0, len(perm), BATCH):
            rows = perm[s:s + BATCH]
            xb = torch.from_numpy(zx(rows))
            ipb = torch.from_numpy(ip_all[rows]); ibb = torch.from_numpy(ib_all[rows])
            yb = torch.from_numpy(yv32[rows])
            cb = torch.from_numpy(cls_t[rows])
            wb = torch.from_numpy(w_all[rows])
            for gparam in opt.param_groups:
                gparam['lr'] = lr_at(step)
            opt.zero_grad()
            ly, lc = net(xb, ipb, ibb)
            loss = (wb * bce(ly, yb)).mean() + 0.5 * (wb * ce(lc, cb)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss); nb += 1; step += 1
        # eval
        net.eval()
        with torch.no_grad():
            pev = torch.sigmoid(net(Xev, ipev, ibev)[0]).numpy().astype(np.float64)
        brier = float(np.mean((pev - yev) ** 2))
        log(f'[{tag}/seed{seed}] ep{ep+1:02d} loss={ep_loss/nb:.5f} evalBrier={brier:.6f}')
        if brier < best[0] - 1e-6:
            best = (brier, {k: v.detach().clone() for k, v in net.state_dict().items()}, ep)
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f'[{tag}/seed{seed}] 얼리스탑 (best ep{best[2]+1}, brier={best[0]:.6f})')
                break
    net.load_state_dict(best[1])
    healthy = best[0] < 0.2560   # 시드건강검사
    if not healthy:
        log(f'[{tag}/seed{seed}] !!! 시드 불건강(brier {best[0]:.4f}) - 폐기')
    net.eval()

    def predict(rows):
        outs = []
        with torch.no_grad():
            for s in range(0, len(rows), 65536):
                r = rows[s:s + 65536]
                xb = torch.from_numpy(zx(r))
                p = torch.sigmoid(net(xb, torch.from_numpy(ip_all[r]), torch.from_numpy(ib_all[r]))[0])
                outs.append(p.numpy().astype(np.float64))
        return np.concatenate(outs)

    return net, predict, healthy, dict(mu=mu, sd=sd, pmap=pmap, bmap=bmap)


# ---------- 1) fold A 학습 ----------
tr_A = season <= 2023
va_A = season == 2024
rows_va = np.where(va_A)[0]
preds_A = []
for seed in (42, 7):
    net, predict, healthy, aux = train_one(tr_A, 2023.0, seed, 'foldA')
    if healthy:
        preds_A.append(predict(rows_va))
assert preds_A, '모든 시드 불건강 - 중단'
p_nn_A = np.mean(preds_A, axis=0)
np.save('dev/cache_nnv2_A.npy', p_nn_A)
log(f'fold A 예측 저장 (건강시드 {len(preds_A)}개 평균)')

# ---------- 2) 스크리닝 ----------
yv = y_all[va_A]
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50']
H = dict(
    base=avg([f'dev/phase90_cache/A_base_{q}.npy' for q in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/A_core_{q}.npy')) *
                    np.load(f'dev/phase90_cache/A_snc_{q}.npy') for q in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/A_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/A_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/A_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/A_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/A_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/A_future50_multi_s{s}.npy' for s in (42, 7)]),
)
W0 = {k: float(v95a[f'{k}_weight']) for k in HEADS8}
t_ = sum(W0.values()); W0 = {k: v / t_ for k, v in W0.items()}
core = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
COMPS = dict(core=core,
             mc6=np.load('dev/cache_mc6head_A.npy'),
             strk=np.load('dev/cache_strk_strk_linear_A.npy'),
             xu=np.load('dev/cache_xgbunused_A.npy'),
             xr=np.load('dev/cache_xgbrawid_A.npy'),
             lty=np.load('dev/cache_lt_y_A.npy'))
W126 = dict(core=0.3491, mc6=0.4381, strk=0.1740, xu=-0.0316, xr=0.0354, lty=0.0350)
blend = np.clip(sum(W126[k] * COMPS[k] for k in COMPS), 0, 1)
resid = yv - blend
E_r2 = float(np.mean(resid ** 2))
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
BASES = [COMPS[k] - blend for k in ('mc6', 'strk', 'xu', 'xr', 'lty')]
BASES.append(np.load('dev/mc6family_cache/A_mc6aux.npy') - blend)


def orth(dd, bases):
    dp = dd - dd.mean()
    for b in bases:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb < 1e-16:
            continue
        dp = dp - (float(np.mean(dp * b)) / Vb) * b
    return dp - dp.mean()


print(f'\nNN단독 BSS = {sc(p_nn_A):.2f}   예측상관(vs blend)={np.corrcoef(p_nn_A, blend)[0,1]:.4f}')
d = p_nn_A - blend
d0 = d - d.mean()
V = float(np.mean(d0 ** 2)); A = float(np.mean(d0 * (blend - yv)))
print(f'원본:    rho={-A/np.sqrt(V*E_r2):+.5f}  V={V:.3e}  s*={-A/V:+.4f}')
dp = orth(d, BASES)
Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
rho_p = -Ap / np.sqrt(Vp * E_r2)
print(f'직교화후: rho={rho_p:+.5f}  이득={K*Ap**2/Vp:+.2f}  s*={-Ap/Vp:+.4f}')
ctrl = []
for sd_ in range(20):
    rng = np.random.RandomState(12000 + sd_)
    dc = orth(rng.permutation(d0), BASES)
    Vc = float(np.mean(dc ** 2))
    if Vc > 1e-18:
        ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
ctrl = np.array(ctrl)
z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
print(f'대조군 z = {z:.1f}  ->  {"통과" if z > 2 else "허수"}')

if z <= 2:
    log('스크리닝 미통과 - 프로덕션 생략, 종료')
    sys.exit(0)

# ---------- 3) 프로덕션 (전체데이터 2시드 + numpy export) ----------
log('스크리닝 통과! 프로덕션 학습 시작...')
tr_full = np.ones(n, dtype=bool)
prod = []
for seed in (42, 7):
    net, predict, healthy, aux = train_one(tr_full, 2024.0, seed, 'prod')
    if healthy:
        sd_np = {k: v.detach().numpy().copy() for k, v in net.state_dict().items()}
        prod.append(dict(state=sd_np, mu=aux['mu'], sd=aux['sd'], pmap=aux['pmap'], bmap=aux['bmap']))
assert prod, '프로덕션 모든 시드 불건강'
joblib.dump(dict(models=prod, feat_order=FEAT, arch='v2_ln_gelu_res512x2_emb8'),
            'dev/nnv2_production.pkl')
log(f'프로덕션 저장 완료 ({len(prod)}시드): dev/nnv2_production.pkl')
