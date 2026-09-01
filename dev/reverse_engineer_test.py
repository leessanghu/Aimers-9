"""리더보드 점수에서 test의 '분리도'를 역산.

Brier = E[p^2] - 2E[py] + r          (y in {0,1} 이므로 y^2=y)
E[p] = m,  Var(p) = v,  E[p^2] = v + m^2
d = E[p|y=1] - E[p|y=0] 라 두면
  E[p|y=1] = m + (1-r)d,  E[p|y=0] = m - r d
  E[py] = r(m + (1-r)d)
=> Brier = v + m^2 - 2r(m + (1-r)d) + r
=> d = [v + m^2 - 2rm + r - Brier] / (2 r (1-r))

필요한 가정: r(test 성공률), m(우리 예측 평균), v(우리 예측 분산).
r은 대회 baseline 상수에서 역산, m/v는 로컬 예측 분포로 근사(감도분석 병행).
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

BASELINE = 0.249807


def solve_d(bss, r, m, v, baseline=BASELINE):
    brier = baseline * (1 - bss / 1e5)
    return (v + m * m - 2 * r * m + r - brier) / (2 * r * (1 - r)), brier


# ---------- 1) 공식 검증: 로컬 fold A에서 실제 d와 일치하는가 ----------
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024; yv = y[va]

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
HARM = ['multires', 'midother', 'condball', 'countresid', 'future50', 'ingame']
W94 = {k: (v * 0.2 if k in HARM else v) for k, v in W.items()}
t = sum(W94.values()); W94 = {k: v / t for k, v in W94.items()}
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
p = 'A'
H = dict(
    base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/{p}_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/{p}_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42, 7)]),
)
P11 = np.load('dev/idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
p94 = sum(W94[k] * H[k] for k in W94)

r_loc = yv.mean(); m_loc = p94.mean(); v_loc = p94.var()
bss_loc = 1e5 * (1 - np.mean((p94 - yv) ** 2) / BASELINE)
d_true = p94[yv == 1].mean() - p94[yv == 0].mean()
d_est, _ = solve_d(bss_loc, r_loc, m_loc, v_loc)
print('=== 공식 검증 (로컬 fold A) ===')
print(f'  BSS={bss_loc:.1f}  r={r_loc:.4f}  m={m_loc:.4f}  std={np.sqrt(v_loc):.4f}')
print(f'  실제 분리도 d = {d_true:+.5f}')
print(f'  공식 역산 d   = {d_est:+.5f}   오차={d_est-d_true:+.2e}')
print()

# ---------- 2) test에 적용 ----------
print('=== test 역산 (r은 baseline에서 두 해 중 하나) ===')
rr = np.roots([1, -1, BASELINE])
print(f'  p(1-p)={BASELINE} 의 해: r = {rr[0]:.4f} 또는 {rr[1]:.4f}')
print()

SUBS = [('v86', 1060.22), ('v88', 1102.83), ('v91', 1097.41), ('v92', 1101.67), ('v93', 1079.74)]
print('가정: 예측 평균 m, 표준편차 s (로컬값 근처로 감도분석)')
print(f'{"제출":>5s} {"BSS":>9s} {"Brier":>9s} | ' + ' | '.join(f'd(s={s:.3f})' for s in (0.045, 0.0527, 0.060)))
for r_use in (0.4861,):
    print(f'-- r={r_use} --')
    for nm, b in SUBS:
        row = []
        for s in (0.045, 0.0527, 0.060):
            d, brier = solve_d(b, r_use, r_use, s * s)
            row.append(f'{d:+.5f}')
        _, brier = solve_d(b, r_use, r_use, 0.0527 ** 2)
        print(f'{nm:>5s} {b:9.2f} {brier:9.6f} | ' + ' | '.join(f'{x:>11s}' for x in row))
print()
print(f'참고: 로컬 fold A 실제 분리도 = {d_true:+.5f}  (예측 std={np.sqrt(v_loc):.4f})')
print()
print('※ m(예측평균)=r 로 가정함(완전 캘리브레이션 시). m이 r보다 크면(과대예측) d는 더 커짐.')
for dm in (0.000, 0.005, 0.010):
    d, _ = solve_d(1102.83, 0.4861, 0.4861 + dm, 0.0527 ** 2)
    print(f'   v88: m-r={dm:+.3f} -> d={d:+.5f}')
