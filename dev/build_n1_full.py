"""N1 프로덕션: 원시컨텍스트53+원시비율18(=71) + ID임베딩4종, 전체데이터 2시드.
fold A z=2.4 통과(nn_raw까지 직교화한 뒤에도 생존). nn_raw를 대체."""
import sys, time, math
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
import torch
import torch.nn as nn

torch.set_num_threads(4)
t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.0f}s] {m}', flush=True)

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
RAW18 = [
    'asof_pitcher_success_rate', 'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate',
    'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate',
    'asof_pitcher_prev1_game_success_rate', 'asof_pitcher_prev3_game_success_rate',
    'asof_pitcher_prev5_game_success_rate', 'asof_pitcher_prev1_game_middle_rate',
    'asof_pitcher_prev3_game_middle_rate', 'asof_pitcher_prev5_game_middle_rate',
    'asof_batter_success_rate', 'asof_batter_middle_rate',
    'asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate',
    'asof_pitcher_n', 'asof_batter_n',
]
ALL_FEATS = CONTEXT_FEATS + RAW18   # 순서 고정 - 추론시 동일 순서 필요

X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y_all = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
n = len(df)
X_ctx = X_df[CONTEXT_FEATS].astype(np.float64).to_numpy()
X_raw18 = df[RAW18].astype(np.float64).to_numpy()
X_all = np.concatenate([X_ctx, X_raw18], axis=1)
pid_raw = df['pitcher_id'].to_numpy(); bid_raw = df['batter_id'].to_numpy()
ptid_raw = df['pitcher_team_id'].to_numpy(); btid_raw = df['batter_team_id'].to_numpy()
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
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


rev = diff_label('asof_pitcher_reverse_rate'); mid = diff_label('asof_pitcher_middle_rate')
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
log('데이터 준비 완료')


class Net(nn.Module):
    def __init__(self, n_feat, n_p, n_b, n_pt, n_bt, hid=512, drop=0.15):
        super().__init__()
        self.emb_p = nn.Embedding(n_p + 1, 8); self.emb_b = nn.Embedding(n_b + 1, 8)
        self.emb_pt = nn.Embedding(n_pt + 1, 4); self.emb_bt = nn.Embedding(n_bt + 1, 4)
        self.emb_drop = nn.Dropout(0.10)
        act = lambda: nn.GELU(approximate='tanh')
        d_in = n_feat + 24
        self.inp = nn.Sequential(nn.Linear(d_in, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.b1 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.b2 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.head_y = nn.Linear(hid, 1); self.head_c = nn.Linear(hid, 6)

    def forward(self, xz, ip, ib, ipt, ibt):
        e = torch.cat([self.emb_drop(self.emb_p(ip)), self.emb_drop(self.emb_b(ib)),
                       self.emb_drop(self.emb_pt(ipt)), self.emb_drop(self.emb_bt(ibt))], dim=1)
        h = self.inp(torch.cat([xz, e], dim=1))
        h = h + self.b1(h); h = h + self.b2(h)
        return self.head_y(h).squeeze(1), self.head_c(h)


def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    idx_all = np.arange(n)
    n_es = int(n * 0.92)
    idx_fit, idx_ev = idx_all[:n_es], idx_all[n_es:]
    mu = np.nanmean(X_all[idx_fit], axis=0); sd = np.nanstd(X_all[idx_fit], axis=0) + 1e-9
    def zx(rows):
        z = (X_all[rows] - mu) / sd
        return np.clip(np.nan_to_num(z, nan=0.0), -10, 10).astype(np.float32)
    maps, idxs = [], []
    for arr in (pid_raw, bid_raw, ptid_raw, btid_raw):
        mp = {v: i + 1 for i, v in enumerate(pd.unique(arr[idx_fit]))}
        maps.append(mp); idxs.append(np.array([mp.get(v, 0) for v in arr], dtype=np.int64))
    w_all = (0.5 ** ((2024.0 - season) / 2.0)).astype(np.float32)
    yv32 = y_all.astype(np.float32)
    net = Net(X_all.shape[1], *[len(m_) for m_ in maps])
    emb_params = [p for nm_, p in net.named_parameters() if nm_.startswith('emb_')]
    other = [p for nm_, p in net.named_parameters() if not nm_.startswith('emb_')]
    opt = torch.optim.AdamW([dict(params=emb_params, weight_decay=3e-2),
                              dict(params=other, weight_decay=1e-4)], lr=1e-3)
    EPOCHS, BATCH = 18, 16384
    steps_per_ep = math.ceil(len(idx_fit) / BATCH); total_steps = EPOCHS * steps_per_ep; warm = steps_per_ep
    def lr_at(step):
        if step < warm: return 1e-3 * (step + 1) / warm
        prog = (step - warm) / max(1, total_steps - warm)
        return 1e-5 + 0.5 * (1e-3 - 1e-5) * (1 + math.cos(math.pi * prog))
    bce = nn.BCEWithLogitsLoss(reduction='none'); ce = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    Xev = torch.from_numpy(zx(idx_ev)); evt = [torch.from_numpy(a[idx_ev]) for a in idxs]; yev = y_all[idx_ev]
    best = (1e9, None, -1); step = 0; patience, bad = 4, 0
    for ep in range(EPOCHS):
        net.train(); perm = np.random.permutation(idx_fit); ep_loss, nb = 0.0, 0
        for s in range(0, len(perm), BATCH):
            rows = perm[s:s + BATCH]
            xb = torch.from_numpy(zx(rows)); bt = [torch.from_numpy(a[rows]) for a in idxs]
            yb = torch.from_numpy(yv32[rows]); cb = torch.from_numpy(cls[rows]); wb = torch.from_numpy(w_all[rows])
            for g_ in opt.param_groups: g_['lr'] = lr_at(step)
            opt.zero_grad()
            ly, lc = net(xb, *bt)
            loss = (wb * bce(ly, yb)).mean() + 0.5 * (wb * ce(lc, cb)).mean()
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            ep_loss += float(loss.detach()); nb += 1; step += 1
        net.eval()
        with torch.no_grad():
            pev = torch.sigmoid(net(Xev, *evt)[0]).numpy().astype(np.float64)
        brier = float(np.mean((pev - yev) ** 2))
        log(f'[prod/seed{seed}] ep{ep+1:02d} loss={ep_loss/nb:.5f} evalBrier={brier:.6f}')
        if brier < best[0] - 1e-6:
            best = (brier, {k2: v.detach().clone() for k2, v in net.state_dict().items()}, ep); bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f'[prod/seed{seed}] 얼리스탑 (best ep{best[2]+1})'); break
    net.load_state_dict(best[1])
    healthy = best[0] < 0.2560
    return net, maps, mu, sd, healthy, best[0]


prod = []
for seed in (42, 7):
    net, maps, mu, sd, healthy, brier = train_one(seed)
    if healthy:
        sd_np = {k2: v.detach().numpy().copy() for k2, v in net.state_dict().items()}
        prod.append(dict(state=sd_np, mu=mu, sd=sd, pmap=maps[0], bmap=maps[1], ptmap=maps[2], btmap=maps[3]))
    else:
        log(f'seed{seed} 불건강(brier={brier:.4f}) - 폐기')
assert prod, '모든 시드 불건강'
joblib.dump(dict(models=prod, feat_order=ALL_FEATS, context_feats=CONTEXT_FEATS, raw18=RAW18,
                  arch='n1_ctx53_raw18_emb8-8-4-4'), 'dev/n1_production.pkl')
log(f'저장 완료 ({len(prod)}시드): dev/n1_production.pkl')
