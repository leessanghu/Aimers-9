"""NN 최종판(NF): 트리가 못 쓰는 재료 총동원 + NN 고유 구조.

피처: 원시컨텍스트53 + 원시비율18 + trackman 32개(중복 tm_n/tm_matched 제외)
아키텍처: ID임베딩 4종 + bilinear 매치업(emb_p ⊙ emb_b 8차원 트렁크 직접투입)
타겟: y(BCE) + mc6 6클래스(CE 0.5) + 연속실패길이 회귀(MSE 0.3, 유효행만)
시드: 3개 평균.
스크리닝: v126 기준, 직교화 축 8개(기존5 + mc6aux + nnraw + N1) + 순열대조군 z.
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

X_df = pd.read_parquet('dev/featcache_X.parquet')
TM = [c for c in X_df.columns if c.startswith('tm_') and c not in ('tm_n', 'tm_matched')]
meta = pd.read_parquet('dev/featcache_meta.parquet')
y_all = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
n = len(df)
X_all = np.concatenate([
    X_df[CONTEXT_FEATS].astype(np.float64).to_numpy(),
    df[RAW18].astype(np.float64).to_numpy(),
    X_df[TM].astype(np.float64).to_numpy(),
], axis=1)
FEATS_ORDER = dict(context=CONTEXT_FEATS, raw18=RAW18, tm=TM)
log(f'피처: ctx {len(CONTEXT_FEATS)} + raw {len(RAW18)} + tm {len(TM)} = {X_all.shape[1]}')

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

# 연속실패길이 (streak) - build_streak_head_experiment와 동일 복원
same_prev = np.zeros(len(order), dtype=bool)
same_prev[1:] = (pid_raw[order][1:] == pid_raw[order][:-1])
dn_ord = np.full(len(order), np.nan)
dn_ord[1:] = n_[order][1:] - n_[order][:-1]
valid_prev = same_prev & (dn_ord == 1)
yo = y_all[order]
streak_ord = np.zeros(len(order))
cur = 0.0
for i in range(len(order)):
    if not valid_prev[i]:
        cur = 0.0
    streak_ord[i] = cur
    cur = 0.0 if yo[i] == 1 else cur + 1
streak = np.empty(n); streak[order] = streak_ord
okm = np.full(n, False); okm[order] = valid_prev
streak_t = np.where(okm, np.clip(streak, 0, 10) / 10.0, np.nan)
log(f'streak 라벨 유효 {np.isfinite(streak_t).mean()*100:.1f}%')


class Net(nn.Module):
    def __init__(self, n_feat, n_p, n_b, n_pt, n_bt, hid=512, drop=0.15):
        super().__init__()
        self.emb_p = nn.Embedding(n_p + 1, 8); self.emb_b = nn.Embedding(n_b + 1, 8)
        self.emb_pt = nn.Embedding(n_pt + 1, 4); self.emb_bt = nn.Embedding(n_bt + 1, 4)
        self.emb_drop = nn.Dropout(0.10)
        act = lambda: nn.GELU(approximate='tanh')
        d_in = n_feat + 24 + 8   # +8 = bilinear(emb_p ⊙ emb_b)
        self.inp = nn.Sequential(nn.Linear(d_in, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.b1 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.b2 = nn.Sequential(nn.Linear(hid, hid), nn.LayerNorm(hid), act(), nn.Dropout(drop))
        self.head_y = nn.Linear(hid, 1)
        self.head_c = nn.Linear(hid, 6)
        self.head_s = nn.Linear(hid, 1)

    def forward(self, xz, ip, ib, ipt, ibt):
        ep = self.emb_drop(self.emb_p(ip)); eb = self.emb_drop(self.emb_b(ib))
        e = torch.cat([ep, eb, self.emb_drop(self.emb_pt(ipt)),
                       self.emb_drop(self.emb_bt(ibt)), ep * eb], dim=1)
        h = self.inp(torch.cat([xz, e], dim=1))
        h = h + self.b1(h); h = h + self.b2(h)
        return self.head_y(h).squeeze(1), self.head_c(h), self.head_s(h).squeeze(1)


def train_one(tr_mask, upto, seed, tag):
    torch.manual_seed(seed); np.random.seed(seed)
    idx_tr_all = np.where(tr_mask)[0]
    n_es = int(len(idx_tr_all) * 0.92)
    idx_fit, idx_ev = idx_tr_all[:n_es], idx_tr_all[n_es:]
    mu = np.nanmean(X_all[idx_fit], axis=0); sd = np.nanstd(X_all[idx_fit], axis=0) + 1e-9
    def zx(rows):
        z = (X_all[rows] - mu) / sd
        return np.clip(np.nan_to_num(z, nan=0.0), -10, 10).astype(np.float32)
    maps, idxs = [], []
    for arr in (pid_raw, bid_raw, ptid_raw, btid_raw):
        mp = {v: i + 1 for i, v in enumerate(pd.unique(arr[idx_fit]))}
        maps.append(mp); idxs.append(np.array([mp.get(v, 0) for v in arr], dtype=np.int64))
    w_all = (0.5 ** ((upto - season) / 2.0)).astype(np.float32)
    yv32 = y_all.astype(np.float32)
    st32 = np.nan_to_num(streak_t, nan=0.0).astype(np.float32)
    st_ok = np.isfinite(streak_t).astype(np.float32)
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
            yb = torch.from_numpy(yv32[rows]); cb = torch.from_numpy(cls[rows])
            wb = torch.from_numpy(w_all[rows])
            sb = torch.from_numpy(st32[rows]); sob = torch.from_numpy(st_ok[rows])
            for g_ in opt.param_groups: g_['lr'] = lr_at(step)
            opt.zero_grad()
            ly, lc, ls = net(xb, *bt)
            loss = (wb * bce(ly, yb)).mean() + 0.5 * (wb * ce(lc, cb)).mean() \
                   + 0.3 * (wb * sob * (ls - sb) ** 2).mean()
            loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            ep_loss += float(loss.detach()); nb += 1; step += 1
        net.eval()
        with torch.no_grad():
            pev = torch.sigmoid(net(Xev, *evt)[0]).numpy().astype(np.float64)
        brier = float(np.mean((pev - yev) ** 2))
        log(f'[{tag}/seed{seed}] ep{ep+1:02d} loss={ep_loss/nb:.5f} evalBrier={brier:.6f}')
        if brier < best[0] - 1e-6:
            best = (brier, {k2: v.detach().clone() for k2, v in net.state_dict().items()}, ep); bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f'[{tag}/seed{seed}] 얼리스탑 (best ep{best[2]+1})'); break
    net.load_state_dict(best[1]); net.eval()
    healthy = best[0] < 0.2560

    def predict(rows):
        outs = []
        with torch.no_grad():
            for s in range(0, len(rows), 65536):
                r = rows[s:s + 65536]
                bt = [torch.from_numpy(a[r]) for a in idxs]
                outs.append(torch.sigmoid(net(torch.from_numpy(zx(r)), *bt)[0]).numpy().astype(np.float64))
        return np.concatenate(outs)
    return net, predict, healthy, dict(mu=mu, sd=sd, maps=maps)


# ---- fold A 3시드 ----
tr_A = season <= 2023
va_A = season == 2024
rows_va = np.where(va_A)[0]
preds = []
for seed in (42, 7, 2024):
    net, predict, healthy, aux = train_one(tr_A, 2023.0, seed, 'foldA')
    if healthy:
        preds.append(predict(rows_va))
assert preds, '모든 시드 불건강'
p_nf = np.mean(preds, axis=0)
np.save('dev/cache_nnfinal_A.npy', p_nf)
log(f'fold A 저장 (건강 {len(preds)}시드)')

# ---- 스크리닝 (직교화 8축) ----
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
W0 = {k2: float(v95a[f'{k2}_weight']) for k2 in HEADS8}
t_ = sum(W0.values()); W0 = {k2: v / t_ for k2, v in W0.items()}
core = np.clip(sum(W0[k2] * H[k2] for k2 in HEADS8), 0, 1)
COMPS = dict(core=core,
             mc6=np.load('dev/cache_mc6head_A.npy'),
             strk=np.load('dev/cache_strk_strk_linear_A.npy'),
             xu=np.load('dev/cache_xgbunused_A.npy'),
             xr=np.load('dev/cache_xgbrawid_A.npy'),
             lty=np.load('dev/cache_lt_y_A.npy'))
W126 = dict(core=0.3491, mc6=0.4381, strk=0.1740, xu=-0.0316, xr=0.0354, lty=0.0350)
blend = np.clip(sum(W126[k2] * COMPS[k2] for k2 in COMPS), 0, 1)
E_r2 = float(np.mean((yv - blend) ** 2))
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
BASES = [COMPS[k2] - blend for k2 in ('mc6', 'strk', 'xu', 'xr', 'lty')]
BASES.append(np.load('dev/mc6family_cache/A_mc6aux.npy') - blend)
BASES.append(np.load('dev/cache_nnraw_A.npy') - blend)
BASES.append(np.load('dev/cache_nn_n1_A.npy') - blend)


def orth(dd, bases):
    dp = dd - dd.mean()
    for b in bases:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb < 1e-16:
            continue
        dp = dp - (float(np.mean(dp * b)) / Vb) * b
    return dp - dp.mean()


print(f'\nNF 단독 BSS = {sc(p_nf):.2f}  예측상관(vs blend)={np.corrcoef(p_nf, blend)[0,1]:.4f}')
d = p_nf - blend; d0 = d - d.mean()
V = float(np.mean(d0 ** 2)); A = float(np.mean(d0 * (blend - yv)))
print(f'원본:    rho={-A/np.sqrt(V*E_r2):+.5f}  s*={-A/V:+.4f}')
dp = orth(d, BASES)
Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
rho_p = -Ap / np.sqrt(Vp * E_r2)
print(f'직교화후(8축): rho={rho_p:+.5f}  이득={K*Ap**2/Vp:+.2f}  s*={-Ap/Vp:+.4f}')
ctrl = []
for sd_ in range(20):
    rng = np.random.RandomState(15000 + sd_)
    dc = orth(rng.permutation(d0), BASES)
    Vc = float(np.mean(dc ** 2))
    if Vc > 1e-18:
        ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
ctrl = np.array(ctrl)
z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
print(f'대조군 z = {z:.1f}  ->  {"통과" if z > 2 else "허수"}')

if z <= 2:
    log('스크리닝 미통과 - 프로덕션 생략')
    sys.exit(0)

# ---- 프로덕션 (전체데이터 3시드) ----
log('통과! 프로덕션 학습...')
tr_full = np.ones(n, dtype=bool)
prod = []
for seed in (42, 7, 2024):
    net, predict, healthy, aux = train_one(tr_full, 2024.0, seed, 'prod')
    if healthy:
        sd_np = {k2: v.detach().numpy().copy() for k2, v in net.state_dict().items()}
        prod.append(dict(state=sd_np, mu=aux['mu'], sd=aux['sd'],
                          pmap=aux['maps'][0], bmap=aux['maps'][1],
                          ptmap=aux['maps'][2], btmap=aux['maps'][3]))
assert prod, '프로덕션 모든 시드 불건강'
joblib.dump(dict(models=prod, context_feats=CONTEXT_FEATS, raw18=RAW18, tm_feats=TM,
                  arch='nf_ctx53_raw18_tm32_bilinear_streakaux'), 'dev/nnfinal_production.pkl')
log(f'프로덕션 저장 ({len(prod)}시드): dev/nnfinal_production.pkl')
