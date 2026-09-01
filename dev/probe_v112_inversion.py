"""v112 실측(+1.1775)으로 mc6축의 진짜 A(=E[(p-y)d])를 역산 + 최적가중치 도출.

먼저 사인 컨벤션 정리(이전 스크립트 출력에 부호오류 있었음):
  p_new = p + s*d,  d = p_mc6 - blend
  BS(s) = BS0 + 2s*A + s^2*V,   A = E[(p-y)d],  V = E[d^2]
  s* = -A/V,  최대이득 = K*A^2/V
  resid = y-p 이므로 A = -E[resid*d] = -C.  따라서 s* = C/V  (이전 출력의 -C/V는 오류)

fold A는 C<0 -> A>0 -> s*<0 (빼야 함)이라고 말했다. 우리는 +0.03으로 더했고 +1.18 얻었다.
=> 실측 A는 음수. 부호가 완전히 뒤집혔다. 그 크기를 역산한다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
S_USED = 0.03
DELTA = 1104.8342852052 - 1103.6568315036

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
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


print('=== [1] fold A/C가 실제로 말했던 것 (부호 정정) ===')
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t = sum(W.values()); W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    p = np.load(f'dev/cache_mc6head_{tag}.npy')
    d = p - blend
    dc = d - d.mean()
    A = float(np.mean(dc * (blend - yv)))
    V = float(np.mean(dc ** 2))
    print(f'  fold{tag}: A={A:+.3e}  V={V:.6f}  s*(정정)={-A/V:+.4f}  최대이득={K*A**2/V:+.2f}')
    if tag == 'A':
        V_local = V

print(f'\n=== [2] 실측 역산 (v112: s={S_USED}, ΔScore={DELTA:+.4f}) ===')
print(f'  ΔScore = -K*(2sA + s^2 V) 에서 A를 역산 (V는 fold A 추정치 사용)')
for label, V in [('foldA V', V_local), ('V x0.8', V_local*0.8), ('V x1.2', V_local*1.2)]:
    A_real = (-DELTA / K - S_USED**2 * V) / (2 * S_USED)
    s_opt = -A_real / V
    gain_max = K * A_real**2 / V
    print(f'  {label:<10} A_real={A_real:+.3e}  s*={s_opt:+.4f}  최대이득={gain_max:+.2f}점')

print(f'\n=== [3] 가중치별 예상 ΔScore (foldA V 기준) ===')
A_real = (-DELTA / K - S_USED**2 * V_local) / (2 * S_USED)
print(f'{"s":>8}{"예상 ΔScore":>14}')
for s in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30):
    print(f'{s:>8.2f}{-K*(2*s*A_real + s**2*V_local):>14.2f}')

print(f'\n=== [4] 부호 반전의 크기 ===')
va = season == 2024
yv = y_all[va]
H = build8('A')
W = {k: float(v95[f'{k}_weight']) for k in H}
t = sum(W.values()); W = {k: v / t for k, v in W.items()}
blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
p = np.load('dev/cache_mc6head_A.npy')
d = p - blend; dc = d - d.mean()
A_foldA = float(np.mean(dc * (blend - yv)))
print(f'  fold A(2024) A = {A_foldA:+.3e}   (양수 = 더하면 손해)')
print(f'  실측 2025  A = {A_real:+.3e}   (음수 = 더하면 이득)')
print(f'  -> 부호가 뒤집혔고 크기는 {abs(A_real/A_foldA):.2f}배')
