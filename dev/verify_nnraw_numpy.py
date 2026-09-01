"""nn_raw numpy forward가 torch forward와 정확히 일치하는지 검증(1000행 샘플)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib, torch, torch.nn as nn
sys.path.insert(0, 'dev')
from nnraw_numpy_forward import forward_nnraw

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
X_raw = X_df[CONTEXT_FEATS].astype(np.float64).to_numpy()
df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id'])
pid_raw = df['pitcher_id'].to_numpy(); bid_raw = df['batter_id'].to_numpy()
ptid_raw = df['pitcher_team_id'].to_numpy(); btid_raw = df['batter_team_id'].to_numpy()

prod = joblib.load('dev/nnraw_production.pkl')
m0 = prod['models'][0]
state = m0['state']
mu, sd = m0['mu'], m0['sd']

rows = np.arange(1000)
z = (X_raw[rows] - mu) / sd
Xz = np.clip(np.nan_to_num(z, nan=0.0), -10, 10).astype(np.float32)
ip = np.array([m0['pmap'].get(v, 0) for v in pid_raw[rows]], dtype=np.int64)
ib = np.array([m0['bmap'].get(v, 0) for v in bid_raw[rows]], dtype=np.int64)
ipt = np.array([m0['ptmap'].get(v, 0) for v in ptid_raw[rows]], dtype=np.int64)
ibt = np.array([m0['btmap'].get(v, 0) for v in btid_raw[rows]], dtype=np.int64)

# numpy forward
state_np = {k: v.astype(np.float64) if v.dtype != np.int64 else v for k, v in state.items()}
p_np = forward_nnraw(Xz.astype(np.float64), ip, ib, ipt, ibt, state_np)

# torch forward (build_nnraw_full.py의 Net과 동일 구조 재구성)
class Net(nn.Module):
    def __init__(self, n_feat, n_p, n_b, n_pt, n_bt, hid=512, drop=0.15):
        super().__init__()
        self.emb_p = nn.Embedding(n_p, 8); self.emb_b = nn.Embedding(n_b, 8)
        self.emb_pt = nn.Embedding(n_pt, 4); self.emb_bt = nn.Embedding(n_bt, 4)
        self.emb_drop = nn.Dropout(0.10)
        act = lambda: nn.GELU(approximate='tanh')
        d_in = n_feat + 8 + 8 + 4 + 4
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

net = Net(len(CONTEXT_FEATS), state['emb_p.weight'].shape[0], state['emb_b.weight'].shape[0],
           state['emb_pt.weight'].shape[0], state['emb_bt.weight'].shape[0])
net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
net.eval()
with torch.no_grad():
    logit, _ = net(torch.from_numpy(Xz), torch.from_numpy(ip), torch.from_numpy(ib),
                    torch.from_numpy(ipt), torch.from_numpy(ibt))
    p_torch = torch.sigmoid(logit).numpy().astype(np.float64)

diff = np.abs(p_np - p_torch)
print(f'최대차이 = {diff.max():.3e}   평균차이 = {diff.mean():.3e}')
assert diff.max() < 1e-5, '불일치! numpy forward 재확인 필요'
print('검증 통과 - numpy forward가 torch와 완전히 일치')
