"""'작은 가중치로 넣으면 도움되지 않냐'를 정확히 계산.

블렌드에 신호 d를 가중치 s로 더하면 (정확한 항등식):
    ΔBS   = 2s*C + s^2*V          C=E[(p-y)d],  V=E[d^2]
    ΔScore= -K*(2s*C + s^2*V)

핵심: 2sC 항은 C의 부호에 따라 +/- 이지만, **s^2*V 항은 항상 손해**다.
따라서 C의 부호를 모르면(E[C]=0) E[ΔScore] = -K*s^2*V < 0 으로 '기대값이 음수'다.
"작아서 안전"이 아니라 "작아서 손해도 작을 뿐" 이고, 방향을 모르면 순손실이다.

fold A/B/C의 실제 C를 써서 정량화한다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
pid_all = df['pitcher_id'].to_numpy()
cs_all = (df['balls_before'].to_numpy() * 4 + df['strikes_before'].to_numpy())
ptype = np.load('dev/recovered_pitch_type.npy')


def build_blend(tag, heads4=False):
    if heads4:
        H = dict(
            base=avg([f'dev/phase90_cache/{tag}_base_{m}.npy' for m in ('d6', 'd8', 'sub')]),
            hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{m}.npy')) *
                            np.load(f'dev/phase90_cache/{tag}_snc_{m}.npy') for m in ('d6', 'd8')], axis=0),
            multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
            ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]))
    else:
        H = dict(
            base=avg([f'dev/phase90_cache/{tag}_base_{m}.npy' for m in ('d6', 'd8', 'sub')]),
            hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{m}.npy')) *
                            np.load(f'dev/phase90_cache/{tag}_snc_{m}.npy') for m in ('d6', 'd8')], axis=0),
            multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
            ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
            midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
            condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
            countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
            future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]))
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    return np.clip(sum(W[k] * H[k] for k in H), 0, 1)


def build_feature(upto, va_mask):
    tr = (season <= upto) & (ptype >= 0)
    mix_tab = pd.DataFrame({'cs': cs_all[tr], 't': ptype[tr]})
    mix_dist = mix_tab.groupby('cs')['t'].value_counts(normalize=True).unstack(fill_value=0)
    for t in range(3):
        if t not in mix_dist.columns:
            mix_dist[t] = 0.0
    mix_dist = mix_dist[[0, 1, 2]]
    gm = mix_tab['t'].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0)
    g = float(y_all[tr].mean())
    ptab = pd.DataFrame({'p': pid_all[tr], 't': ptype[tr], 'y': y_all[tr]})
    pr = ptab.groupby(['p', 't'])['y'].agg(['sum', 'count'])
    pr['rate'] = (pr['sum'] + 60.0 * g) / (pr['count'] + 60.0)
    rw = pr['rate'].unstack()
    for t in range(3):
        if t not in rw.columns:
            rw[t] = g
    rw = rw[[0, 1, 2]].fillna(g)
    mix_row = mix_dist.reindex(cs_all[va_mask]).fillna(gm).to_numpy(np.float64)
    rate_row = rw.reindex(pid_all[va_mask]).fillna(g).to_numpy(np.float64)
    return (mix_row * rate_row).sum(axis=1)


print('=== fold별 C, V (d = 피처를 블렌드 스케일로 중심화한 것) ===')
print(f'{"fold":<6}{"C":>13}{"V":>12}{"최적 s*":>10}{"최적시 이득":>12}')
CV = []
for tag, upto, vs, h4 in [('A', 2023, 2024, False), ('B', 2022, 2023, True), ('C', 2021, 2022, False)]:
    va = season == vs
    yv = y_all[va]
    blend = build_blend(tag, heads4=h4)
    feat = build_feature(upto, va)
    d = feat - feat.mean()
    C = float(np.mean((blend - yv) * d))
    V = float(np.mean(d ** 2))
    CV.append((tag, C, V))
    print(f'{tag:<6}{C:>+13.3e}{V:>12.6f}{-C/V:>+10.4f}{K*C**2/V:>+12.2f}')

print('\n=== ΔScore(s) = -K*(2sC + s^2 V)  — fold별 및 평균 ===')
print(f'{"s":>8}' + ''.join(f'{f"fold{t}":>12}' for t, _, _ in CV) + f'{"평균":>12}{"s^2V 손실항":>14}')
for s in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
    vals = []
    for tag, C, V in CV:
        vals.append(-K * (2 * s * C + s ** 2 * V))
    Cbar = np.mean([c for _, c, _ in CV])
    Vbar = np.mean([v for _, _, v in CV])
    ev = -K * (2 * s * Cbar + s ** 2 * Vbar)
    penalty = -K * (s ** 2 * Vbar)
    print(f'{s:>8.3f}' + ''.join(f'{v:>+12.2f}' for v in vals) + f'{ev:>+12.2f}{penalty:>+14.3f}')

print('\n[핵심] 마지막 열 s^2V는 C와 무관하게 항상 음수(손해).')
print(' C의 부호를 모르면 기대값 = 그 손실항만 남는다 -> 어떤 s>0도 기대값 음수.')
Cbar = np.mean([c for _, c, _ in CV])
Vbar = np.mean([v for _, _, v in CV])
Cs = [c for _, c, _ in CV]
print(f'\n 3-fold 평균 C = {Cbar:+.3e}  (부호 A:- B:- C:+ 로 불일치)')
print(f' fold간 C 표준편차 = {np.std(Cs):.3e}  <- 평균값 크기의 {abs(np.std(Cs)/Cbar):.1f}배')
print(f' 즉 "평균이 음수"라는 것조차 fold간 산포에 묻힘 -> 방향 미지 상태 확정.')
