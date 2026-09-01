"""nn_raw 추론용 순수 numpy forward. torch 불필요.
build_nnraw_full.py의 Net 아키텍처(LN+GELU+residual x2)를 numpy로 재구현.
state_dict 키 이름과 완전히 일치해야 함(nn.Sequential 인덱스 기반).
"""
import numpy as np


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x ** 3)))


def layernorm(x, w, b, eps=1e-5):
    mu = x.mean(axis=1, keepdims=True)
    var = x.var(axis=1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * w + b


def linear(x, w, b):
    return x @ w.T + b


def block(x, state, prefix):
    # Sequential: [0]=Linear [1]=LayerNorm [2]=GELU [3]=Dropout(eval에서 identity)
    h = linear(x, state[f'{prefix}.0.weight'], state[f'{prefix}.0.bias'])
    h = layernorm(h, state[f'{prefix}.1.weight'], state[f'{prefix}.1.bias'])
    h = gelu_tanh(h)
    return h


def forward_nnraw(Xz, ip, ib, ipt, ibt, state):
    """Xz: (n, 52) 표준화된 원시피처. ip/ib/ipt/ibt: (n,) int 인덱스(0=미등록)."""
    ep = state['emb_p.weight'][ip]
    eb = state['emb_b.weight'][ib]
    ept = state['emb_pt.weight'][ipt]
    ebt = state['emb_bt.weight'][ibt]
    # eval 모드 -> emb_drop은 identity
    e = np.concatenate([ep, eb, ept, ebt], axis=1)
    x = np.concatenate([Xz, e], axis=1)
    h = block(x, state, 'inp')
    h = h + block(h, state, 'b1')
    h = h + block(h, state, 'b2')
    logit = linear(h, state['head_y.weight'], state['head_y.bias']).squeeze(1)
    return 1.0 / (1.0 + np.exp(-logit))
