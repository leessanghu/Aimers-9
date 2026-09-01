"""N1 numpy forward가 torch와 정확히 일치하는지 검증(1000행)."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib, torch, torch.nn as nn
sys.path.insert(0, 'dev')
from nnraw_numpy_forward import forward_nnraw

prod = joblib.load('dev/n1_production.pkl')
CONTEXT_FEATS = prod['context_feats']; RAW18 = prod['raw18']; ALL_FEATS = prod['feat_order']
assert ALL_FEATS == CONTEXT_FEATS + RAW18

X_df = pd.read_parquet('dev/featcache_X.parquet')
df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['pitcher_id', 'batter_id', 'pitcher_team_id', 'batter_team_id'] + RAW18)
X_ctx = X_df[CONTEXT_FEATS].astype(np.float64).to_numpy()
X_raw18 = df[RAW18].astype(np.float64).to_numpy()
X_all = np.concatenate([X_ctx, X_raw18], axis=1)
pid_raw = df['pitcher_id'].to_numpy(); bid_raw = df['batter_id'].to_numpy()
ptid_raw = df['pitcher_team_id'].to_numpy(); btid_raw = df['batter_team_id'].to_numpy()

m0 = prod['models'][0]; state = m0['state']; mu, sd = m0['mu'], m0['sd']
rows = np.arange(1000)
z = (X_all[rows] - mu) / sd
Xz = np.clip(np.nan_to_num(z, nan=0.0), -10, 10).astype(np.float32)
ip = np.array([m0['pmap'].get(v, 0) for v in pid_raw[rows]], dtype=np.int64)
ib = np.array([m0['bmap'].get(v, 0) for v in bid_raw[rows]], dtype=np.int64)
ipt = np.array([m0['ptmap'].get(v, 0) for v in ptid_raw[rows]], dtype=np.int64)
ibt = np.array([m0['btmap'].get(v, 0) for v in btid_raw[rows]], dtype=np.int64)

state_np = {k: v for k, v in state.items()}
p_np = forward_nnraw(Xz.astype(np.float64), ip, ib, ipt, ibt, state_np)

class Net(nn.Module):
    def __init__(self, n_feat, n_p, n_b, n_pt, n_bt, hid=512, drop=0.15):
        super().__init__()
        self.emb_p = nn.Embedding(n_p, 8); self.emb_b = nn.Embedding(n_b, 8)
        self.emb_pt = nn.Embedding(n_pt, 4); self.emb_bt = nn.Embedding(n_bt, 4)
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

net = Net(len(ALL_FEATS), state['emb_p.weight'].shape[0], state['emb_b.weight'].shape[0],
           state['emb_pt.weight'].shape[0], state['emb_bt.weight'].shape[0])
net.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
net.eval()
with torch.no_grad():
    logit, _ = net(torch.from_numpy(Xz), torch.from_numpy(ip), torch.from_numpy(ib),
                    torch.from_numpy(ipt), torch.from_numpy(ibt))
    p_torch = torch.sigmoid(logit).numpy().astype(np.float64)

diff = np.abs(p_np - p_torch)
print(f'최대차이 = {diff.max():.3e}   평균차이 = {diff.mean():.3e}')
assert diff.max() < 1e-5, '불일치!'
print('검증 통과')
