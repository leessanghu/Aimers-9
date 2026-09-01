"""v108 실측(1102.4631476375) 1회로 XGB raw-ID 헤드의 '진짜' 실측 신호를 역산.

항등식(정확):
  p_v108 = p_v95 + s*d,   d = p_xgb - p_v95_rawblend,  s = 0.03
  BS(s) = BS0 + 2s*C + s^2*V,   C = E[(p_v95-y)*d],  V = E[d^2]
  ΔScore = -K*(2sC + s^2 V),    K = 1e5/0.249807

V(=E[d^2])는 분포량이라 fold A/C에서 잘 추정됨 -> 실측 ΔScore로 C를 역산.
그러면 최적가중치 s* = -C/V, 최대이득 = K*C^2/V 가 나온다.
이게 양수로 크면 XGB 튜닝/가중치조정이 의미있고, 0 근처면 축 자체가 닫힌 것.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
S = 0.03
SCORE_V95 = 1103.6568315036
SCORE_V108 = 1102.4631476375
DELTA = SCORE_V108 - SCORE_V95

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


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


print('=== V=E[d^2] 로컬 추정 (fold A/C) ===')
Vs = {}
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t = sum(W.values())
    W = {k: v / t for k, v in W.items()}          # 8헤드로 재정규화 (mc5/ingame 캐시 부재 -> 근사)
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    p_xgb = np.load(f'dev/cache_xgbrawid_{tag}.npy')
    d = p_xgb - blend
    V = float(np.mean(d ** 2))
    C_loc = float(np.mean((blend - yv) * d))
    Vs[tag] = V
    print(f'  fold{tag}: V={V:.6f}  sd(d)={np.sqrt(V):.4f}  |  로컬 C={C_loc:+.3e}'
          f'  로컬s*={-C_loc/V:+.4f}  로컬최대이득={K*C_loc**2/V:+.2f}점')

print(f'\n=== 실측 역산 (ΔScore={DELTA:+.4f}, s={S}) ===')
print(f'{"V가정":>10} {"C(역산)":>12} {"s*(최적)":>10} {"최대이득":>10} {"s=0.03이득":>11}')
for label, V in [('foldA', Vs['A']), ('foldC', Vs['C']), ('평균', (Vs['A'] + Vs['C']) / 2),
                 ('foldA×0.8', Vs['A'] * 0.8), ('foldA×1.2', Vs['A'] * 1.2)]:
    # -K*(2sC + s^2 V) = DELTA  ->  C = (-DELTA/K - s^2 V) / (2s)
    C = (-DELTA / K - S ** 2 * V) / (2 * S)
    s_opt = -C / V
    gain_max = K * C ** 2 / V
    gain_003 = -K * (2 * S * C + S ** 2 * V)
    print(f'{label:>10} {C:+12.3e} {s_opt:+10.4f} {gain_max:+10.2f} {gain_003:+11.2f}')

print('\n[해석]')
print(' - s*가 0 근처(<0.01)면 XGB축은 이미 닫힘 -> 튜닝 무의미')
print(' - 최대이득이 +2점 미만이면 필요한 +20점 대비 무의미')
print(' - 로컬C vs 역산C의 부호/크기 차이 = local-weight-optimization-trap 크기')
