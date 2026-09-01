"""구종을 타겟이 아니라 '파생피처'로 사용. 아이디어(사용자):
  expected_success_by_mix(row) = sum_t P(구종=t | count, 투수) * 투수의 (구종=t일때 성공률)

두 요소:
  1) P(구종=t | count) : 전역 카운트별 구종분포 (train 통계, causal)
  2) 투수별 (구종=t일때 성공률) : walk-forward 축소평균 (Rule4 안전, 과거만 사용)

기존 asof_pitcher_fastball_rate(시즌누적 구종비율)에는 없는 신호:
  - 카운트로 조건화된 '이번 투구 예상 구종분포'
  - 구종별 성공률 교차(투수가 특정구종에 유독 강한/약한 것)

이게 v95 잔차와 상관있는지 fold A/C에서 honest하게 검증(중심화+대조군).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
NEED_RHO = 0.01740

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
pid = df['pitcher_id'].to_numpy()
cs = (df['balls_before'].to_numpy() * 4 + df['strikes_before'].to_numpy())
ptype = np.load('dev/recovered_pitch_type.npy')   # 0=직구 1=변화구 2=오프스피드, -1=복원불가

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{m}.npy' for m in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{m}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{m}.npy') for m in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


def build_feature(upto, va_mask):
    """train<=upto 데이터로만 통계 -> va_mask에 적용 (walk-forward, Rule4 안전)"""
    tr = (season <= upto) & (ptype >= 0)

    # (1) 카운트별 전역 구종분포 P(t|count)
    mix_tab = pd.DataFrame({'cs': cs[tr], 't': ptype[tr]})
    mix_dist = mix_tab.groupby('cs')['t'].value_counts(normalize=True).unstack(fill_value=0)
    for t in range(3):
        if t not in mix_dist.columns:
            mix_dist[t] = 0.0
    mix_dist = mix_dist[[0, 1, 2]]
    global_mix = mix_tab['t'].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0)

    # (2) 투수별 구종별 성공률 (축소, K=60)
    g = float(y[tr].mean())
    ptab = pd.DataFrame({'p': pid[tr], 't': ptype[tr], 'y': y[tr]})
    p_rate = ptab.groupby(['p', 't'])['y'].agg(['sum', 'count'])
    K_SH = 60.0
    p_rate['rate'] = (p_rate['sum'] + K_SH * g) / (p_rate['count'] + K_SH)
    rate_wide = p_rate['rate'].unstack()
    for t in range(3):
        if t not in rate_wide.columns:
            rate_wide[t] = g
    rate_wide = rate_wide[[0, 1, 2]].fillna(g)

    # va_mask 행에 적용
    cs_va = cs[va_mask]
    pid_va = pid[va_mask]
    mix_row = mix_dist.reindex(cs_va).fillna(global_mix).to_numpy(np.float64)   # (n,3)
    rate_row = rate_wide.reindex(pid_va).fillna(g).to_numpy(np.float64)          # (n,3)
    return (mix_row * rate_row).sum(axis=1)   # expected_success_by_mix


for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend

    feat = build_feature(upto, va)
    d = feat - feat.mean()
    V = float(np.mean(d ** 2)); C = float(np.mean(d * resid))
    rho = C / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
    print(f'\n=== fold {tag} ({vs}) ===')
    print(f'  expected_success_by_mix 분포: mean={feat.mean():.4f} std={feat.std():.4f}')
    print(f'  잔차상관 rho = {rho:+.5f}   (+30점 필요치 {NEED_RHO:.5f}의 {abs(rho)/NEED_RHO*100:.1f}%)')
    print(f'  최대이득 = {K*C**2/V if V>1e-14 else 0:+.2f}점')
