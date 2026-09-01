"""NN raw: 가공피처(축소평균 등) 전부 제외, 원시컨텍스트 52 + ID임베딩 4종만으로 학습.

가설(사용자): NN에게 우리 162피처(우리가 만든 표현)를 주지 말고 원시입력만 줘서
  스스로 표현을 만들게 하면, 트리앙상블이 못 보는 걸 볼 수 있다.
사전증거는 반대방향(xgb_ctx 잔차상관 불변, ICC 바닥, nnv2 z=0.6)이나
  'NN+원시입력'은 매트릭스의 마지막 미검증 칸이라 확인 가치 있음.

아키텍처/학습: build_nn_v2.py와 동일 레시피(검증된 안정화). 차이:
  - 트렁크 입력: CONTEXT_FEATS 52개 (xgb_ctx와 동일 목록 - 비교가능성)
  - 임베딩: 투수8 + 타자8 + 투수팀4 + 타자팀4
스크리닝: v126 기준 6축 직교화 + 순열대조군 z. z>2일 때만 프로덕션.
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

CONTEXT_FEATS = [
    'cat_top_bottom', 'cat_game_type', 'cat_base_state', 'season', 'game_month',
    'game_dayofweek', 'inning', 'balls_before', 'strikes_before', 'outs_before',
    'run_top_before', 'run_bot_before', 'run_total_before', 'score_diff_home',
    'score_diff_pitcher_team', 'runner_on_1b', 'runner_on_2b', 'runner_on_3b',
    'num_runners_on', 'home_win_expectancy', 'away_win_expectancy', 'li',
    'pitcher_hand', 'batter_hand', 'same_hand', 'count_state', 'hand_matchup',
    'flag_asof_pitcher_n_zero', 'asof_pitcher_n', 'flag_asof_batter_n_zero',
    'asof_batter_n', 'flag_asof_pitcher_pitchmix_n_zero', 'asof_pitcher_pitchmix_n',
    'flag_prev_game_missing', 'pitcher_id_count', 'batter_id_count',
    'pitcher_team_id_count', 'batter_team_id_count', 'inseason_n',
    'inseason_is_first_appearance', 'platoon_n', 'inning_n', 'pt_n',
    'x_count_pressure', 'count_n', 'vol_n_seasons', 'role_n_app', 'form_missing',
    'tm_n', 'tm_matched', 'bat_inseason_n', 'bat_ly_n', 'bplatoon_n',
]

X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
missing = [c for c in CONTEXT_FEATS if c not in X_df.columns]
assert not missing, f'없는 컬럼: {missing}'
y_all = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
X_raw = X_df[CONTEXT_FEATS].astype(np.float64).to_numpy()
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id',
                          'asof_pitcher_n', 'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid_raw = df['pitcher_id'].to_numpy()
bid_raw = df['batter_id'].to_numpy()
ptid_raw = df['pitcher_team_id'].to_numpy()
btid_raw = df['batter_team_id'].to_numpy()
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
ball = call[:, 0]; strike = call[:, 1]; inplay = call[:, 2]
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


class Net(nn.Module):
    def __init__(self, n_feat, n_p, n_b, n_pt, n_bt, hid=512, drop=0.15):
        super().__init__()
        self.emb_p = nn.Embedding(n_p + 1, 8)
        self.emb_b = nn.Embedding(n_b + 1, 8)
        self.emb_pt = nn.Embedding(n_pt + 1, 4)
        self.emb_bt = nn.Embedding(n_bt + 1, 4)
        self.emb_drop = nn.Dropout(0.10)
        act = lambda: nn.GELU(approximate='tanh')
        d_in = n_feat + 8 + 8 + 4 + 4
        self.inp = nn.Sequential(nn.Linear(d_in, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.b1 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.b2 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.head_y = nn.Linear(hid, 1)
        self.head_c = nn.Linear(hid, 6)

    def forward(self, xz, ip, ib, ipt, ibt):
        e = torch.cat([self.emb_drop(self.emb_p(ip)), self.emb_drop(self.emb_b(ib)),
                       self.emb_drop(self.emb_pt(ipt)), self.emb_drop(self.emb_bt(ibt))], dim=1)
        h = self.inp(torch.cat([xz, e], dim=1))
        h = h + self.b1(h)
        h = h + self.b2(h)
        return self.head_y(h).squeeze(1), self.head_c(h)


def train_one(tr_mask, upto, seed, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    idx_tr_all = np.where(tr_mask)[0]
    n_es = int(len(idx_tr_all) * 0.92)
    idx_fit, idx_ev = idx_tr_all[:n_es], idx_tr_all[n_es:]

    mu = np.nanmean(X_raw[idx_fit], axis=0)
    sd = np.nanstd(X_raw[idx_fit], axis=0) + 1e-9
    def zx(rows):
        z = (X_raw[rows] - mu) / sd
        return np.clip(np.nan_to_num(z, nan=0.0), -10, 10).astype(np.float32)

    maps = []
    idxs = []
    for arr in (pid_raw, bid_raw, ptid_raw, btid_raw):
        mp = {v: i + 1 for i, v in enumerate(pd.unique(arr[idx_fit]))}
        maps.append(mp)
        idxs.append(np.array([mp.get(v, 0) for v in arr], dtype=np.int64))
    ip_all, ib_all, ipt_all, ibt_all = idxs

    w_all = (0.5 ** ((upto - season) / 2.0)).astype(np.float32)
    yv32 = y_all.astype(np.float32)

    net = Net(X_raw.shape[1], len(maps[0]), len(maps[1]), len(maps[2]), len(maps[3]))
    emb_params = [p for nm_, p in net.named_parameters() if nm_.startswith('emb_')]
    other = [p for nm_, p in net.named_parameters() if not nm_.startswith('emb_')]
    opt = torch.optim.AdamW([
        dict(params=emb_params, weight_decay=3e-2),
        dict(params=other, weight_decay=1e-4)], lr=1e-3)

    EPOCHS, BATCH = 18, 16384
    steps_per_ep = math.ceil(len(idx_fit) / BATCH)
    total_steps = EPOCHS * steps_per_ep
    warm = steps_per_ep
    def lr_at(step):
        if step < warm:
            return 1e-3 * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * prog))

    bce = nn.BCEWithLogitsLoss(reduction='none')
    ce = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    Xev = torch.from_numpy(zx(idx_ev))
    evt = [torch.from_numpy(a[idx_ev]) for a in idxs]
    yev = y_all[idx_ev]

    best = (1e9, None, -1); step = 0; patience, bad = 4, 0
    for ep in range(EPOCHS):
        net.train()
        perm = np.random.permutation(idx_fit)
        ep_loss, nb = 0.0, 0
        for s in range(0, len(perm), BATCH):
            rows = perm[s:s + BATCH]
            xb = torch.from_numpy(zx(rows))
            bt = [torch.from_numpy(a[rows]) for a in idxs]
            yb = torch.from_numpy(yv32[rows]); cb = torch.from_numpy(cls[rows])
            wb = torch.from_numpy(w_all[rows])
            for g_ in opt.param_groups:
                g_['lr'] = lr_at(step)
            opt.zero_grad()
            ly, lc = net(xb, *bt)
            loss = (wb * bce(ly, yb)).mean() + 0.5 * (wb * ce(lc, cb)).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.detach()); nb += 1; step += 1
        net.eval()
        with torch.no_grad():
            pev = torch.sigmoid(net(Xev, *evt)[0]).numpy().astype(np.float64)
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
    healthy = best[0] < 0.2560
    if not healthy:
        log(f'[{tag}/seed{seed}] !!! 시드 불건강 - 폐기')
    net.eval()

    def predict(rows):
        outs = []
        with torch.no_grad():
            for s in range(0, len(rows), 65536):
                r = rows[s:s + 65536]
                bt = [torch.from_numpy(a[r]) for a in idxs]
                p = torch.sigmoid(net(torch.from_numpy(zx(r)), *bt)[0])
                outs.append(p.numpy().astype(np.float64))
        return np.concatenate(outs)

    return net, predict, healthy


# fold A
tr_A = season <= 2023
va_A = season == 2024
rows_va = np.where(va_A)[0]
preds = []
for seed in (42, 7):
    net, predict, healthy = train_one(tr_A, 2023.0, seed, 'foldA')
    if healthy:
        preds.append(predict(rows_va))
assert preds, '모든 시드 불건강'
p_nn = np.mean(preds, axis=0)
np.save('dev/cache_nnraw_A.npy', p_nn)
log(f'fold A 저장 (건강시드 {len(preds)}개)')

# 스크리닝
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
E_r2 = float(np.mean((yv - blend) ** 2))
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
BASES = [COMPS[k] - blend for k in ('mc6', 'strk', 'xu', 'xr', 'lty')]
BASES.append(np.load('dev/mc6family_cache/A_mc6aux.npy') - blend)
BASES.append(np.load('dev/cache_nnv2_A.npy') - blend)   # 어제 NN(가공피처판)에도 직교화


def orth(dd, bases):
    dp = dd - dd.mean()
    for b in bases:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb < 1e-16:
            continue
        dp = dp - (float(np.mean(dp * b)) / Vb) * b
    return dp - dp.mean()


print(f'\nNNraw 단독 BSS = {sc(p_nn):.2f}   예측상관(vs blend)={np.corrcoef(p_nn, blend)[0,1]:.4f}')
d = p_nn - blend
d0 = d - d.mean()
V = float(np.mean(d0 ** 2)); A = float(np.mean(d0 * (blend - yv)))
print(f'원본:    rho={-A/np.sqrt(V*E_r2):+.5f}  V={V:.3e}  s*={-A/V:+.4f}')
dp = orth(d, BASES)
Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
rho_p = -Ap / np.sqrt(Vp * E_r2)
print(f'직교화후: rho={rho_p:+.5f}  이득={K*Ap**2/Vp:+.2f}  s*={-Ap/Vp:+.4f}')
ctrl = []
for sd_ in range(20):
    rng = np.random.RandomState(13000 + sd_)
    dc = orth(rng.permutation(d0), BASES)
    Vc = float(np.mean(dc ** 2))
    if Vc > 1e-18:
        ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
ctrl = np.array(ctrl)
z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
print(f'대조군 z = {z:.1f}  ->  {"통과" if z > 2 else "허수"}')
log('완료')
