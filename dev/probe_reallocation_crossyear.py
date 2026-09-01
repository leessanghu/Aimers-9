"""재배분(overlap4->indep3) 가설을 교차연도 이식으로 재검증.
[[stacking-closed-and-crossfold-method]]가 확립한 우월한 방법: 반쪼개기(H1/H2)가 아니라
fold A 전체(2024)에서 최적 t를 학습 -> fold C(2022)에 그대로 이식, 반대도 동일.
재학습 불필요(캐시만 사용).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
OVERLAP = ['midother', 'condball', 'countresid', 'future50']
INDEP = ['base', 'hurdle', 'ordinal']
S_MC6, S_STRK = 0.48, 0.10


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


def make_blend(H, W8, tag):
    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    base8 = sum(W8[k] * H[k] for k in HEADS8)
    return np.clip(rest * base8 + S_MC6 * p_mc6 + S_STRK * p_strk, 0, 1)


def weights_at_t(W0, t):
    Wt = dict(W0)
    overlap_sum = sum(W0[k] for k in OVERLAP)
    indep_sum = sum(W0[k] for k in INDEP)
    move = overlap_sum * t
    for k in OVERLAP:
        Wt[k] = W0[k] * (1 - t)
    for k in INDEP:
        Wt[k] = W0[k] + move * (W0[k] / indep_sum)
    return Wt


W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
t0 = sum(W0.values()); W0 = {k: v / t0 for k, v in W0.items()}

data = {}
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    H = build8(tag)
    blend0 = make_blend(H, W0, tag)
    sc = (lambda yv_: lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv_) ** 2) / B_))(yv)
    data[tag] = dict(H=H, yv=yv, blend0=blend0, sc=sc)

print('=== 각 fold에서 "최적 t"를 그 fold 전체로 찾으면? (참고용, 이식 전) ===')
for tag in ('A', 'C'):
    H, yv, blend0, sc = data[tag]['H'], data[tag]['yv'], data[tag]['blend0'], data[tag]['sc']
    best_t, best_g = 0, 0
    for t in np.linspace(0, 0.6, 13):
        blend_t = make_blend(H, weights_at_t(W0, t), tag)
        g = sc(blend_t) - sc(blend0)
        if g > best_g:
            best_g, best_t = g, t
    print(f'  fold{tag}: 자체최적 t={best_t:.2f}  자체이득={best_g:+.2f}')

print('\n=== 교차연도 이식: fold A에서 찾은 t를 fold C에, 반대도 ===')
print(f'{"t(고정)":>10}{"A에서이득":>12}{"C에서이득":>12}{"평균":>10}')
for t in (0.0, 0.10, 0.20, 0.30, 0.40, 0.50):
    gA = data['A']['sc'](make_blend(data['A']['H'], weights_at_t(W0, t), 'A')) - data['A']['sc'](data['A']['blend0'])
    gC = data['C']['sc'](make_blend(data['C']['H'], weights_at_t(W0, t), 'C')) - data['C']['sc'](data['C']['blend0'])
    print(f'{t:>10.2f}{gA:>+12.2f}{gC:>+12.2f}{(gA+gC)/2:>+10.2f}')

print('\n[핵심] "fold A에서 t를 정하면 fold C에서도 좋은가"(그리고 반대도)가 진짜 검증.')
print(' 위 표는 이미 "같은 t를 양쪽에 적용"이라 이식 테스트와 동일함(t는 fold 무관 상수이므로).')
print(' t가 커질수록 두 열이 같이 커지면(둘 다 양수, 비슷한 크기로) -> 진짜 신호.')
print(' 한쪽만 크거나 부호가 갈리면 -> fold특유 패턴(신뢰 불가).')
