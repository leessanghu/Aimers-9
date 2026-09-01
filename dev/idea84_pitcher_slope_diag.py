"""투수x카운트압박 random slope 진단 (경량, 모델재학습 없음).
가설: 어떤 투수는 2스트라이크(위기 카운트)에서 자기 평균보다 더 잘/못 던진다.
기존 count_diff(K=880)는 과도축소라 이 신호가 거의 0으로 눌려있을 수 있다.

Rule4 안전 구조: 각 행은 그 투수의 '직전까지 누적'만 참조 (asof 패턴과 동일).
검증: fold A(2024) 안에서 H1<->H2 양방향, train은 season<=2023 누적통계로 구성.
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'season', 'pitcher_id', 'balls_before', 'strikes_before',
                          'asof_pitcher_n', 'asof_pitcher_success_rate', 'control_success'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)
df['is_2k'] = (df['strikes_before'] == 2).astype(int)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
assert len(df) == len(X)
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
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
v88_raw = sum(W[k] * H[k] for k in W)
sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)

# ---- 투수x2스트라이크 누적통계 (직전 시즌까지, asof 패턴) ----
train = df[df.season <= 2023]
g = train.groupby(['pitcher_id', 'is_2k'])['control_success'].agg(['sum', 'count']).unstack(fill_value=0)
s2 = g[('sum', 1)]; n2 = g[('sum', 1)] * 0 + g.get(('count', 1), 0)
n2 = g[('count', 1)] if ('count', 1) in g.columns else pd.Series(0, index=g.index)
s0 = g[('sum', 0)] if ('sum', 0) in g.columns else pd.Series(0, index=g.index)
n0 = g[('count', 0)] if ('count', 0) in g.columns else pd.Series(0, index=g.index)
rate2 = (s2 / n2.replace(0, np.nan))
rate0 = (s0 / n0.replace(0, np.nan))
global_gap = (train[train.is_2k == 1].control_success.mean() - train[train.is_2k == 0].control_success.mean())
print(f'전역 2스트라이크 갭 (train<=2023) = {global_gap:+.5f}')
print(f'투수별 n2 분포: median={n2.median():.0f} mean={n2.mean():.0f}')
print()

va_idx = df.index[va]
pid_va = df.loc[va_idx, 'pitcher_id'].to_numpy()
n2_va = n2.reindex(pid_va).fillna(0).to_numpy(np.float64)
n0_va = n0.reindex(pid_va).fillna(0).to_numpy(np.float64)
rate2_va = rate2.reindex(pid_va).to_numpy(np.float64)
rate0_va = rate0.reindex(pid_va).to_numpy(np.float64)
raw_gap = np.nan_to_num(rate2_va - rate0_va, nan=0.0)

H1 = mth <= 6; H2 = ~H1
resid_base = yv - v88_raw


def eval_slope(gap_shrunk, weight_n):
    """gap_shrunk: 축소된 슬로프값. weight_n: 신뢰도(작을수록 0으로). threshold-free, 직접 additive."""
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        center = gap_shrunk[fit_m].mean()
        cc = gap_shrunk - center
        C = np.mean(cc[fit_m] * resid_base[fit_m])
        V = np.mean(cc[fit_m] ** 2)
        a = C / V if V > 1e-12 else 0.0
        adj = v88_raw.copy(); adj[ev_m] = v88_raw[ev_m] + a * cc[ev_m]
        gains.append(sc(adj, ev_m) - sc(v88_raw, ev_m))
    return gains, a


print('=== K(축소강도)별 슬로프 신호 세기 ===')
for K in [10, 20, 50, 100, 200, 400, 880]:
    shrunk = raw_gap * (n2_va / (n2_va + K))
    # is_2k 여부에 따라 실제 적용되는 correction (그 행이 2K면 gap, 아니면 0 -> 2K에서만 적용)
    is2k_va = df.loc[va_idx, 'is_2k'].to_numpy()
    applied = np.where(is2k_va == 1, shrunk, 0.0)
    gains, a_last = eval_slope(applied, None)
    print(f'  K={K:4d}  적용된 shrunk 슬로프 std={applied.std():.5f}  H1->H2={gains[0]:+7.2f}  H2->H1={gains[1]:+7.2f}  평균={np.mean(gains):+7.2f}')
