"""'fold C가 힘든 해라 거기서 나온 신호가 더 믿을만하다'를 검증.
투수단위 부트스트랩으로 두 fold의 rho에 대한 실제 SE를 재서 비교한다.
"힘들다"(절대점수가 낮다/불안정하다)가 "노이즈가 작다"를 보장하지 않는다 -
오히려 노이즈가 큰 해일수록 우연히 큰 rho가 나올 확률도 커진다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
pid_all = df['pitcher_id'].to_numpy()
cs_all = (df['balls_before'].to_numpy() * 4 + df['strikes_before'].to_numpy())
ptype = np.load('dev/recovered_pitch_type.npy')


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
    tr = (season <= upto) & (ptype >= 0)
    mix_tab = pd.DataFrame({'cs': cs_all[tr], 't': ptype[tr]})
    mix_dist = mix_tab.groupby('cs')['t'].value_counts(normalize=True).unstack(fill_value=0)
    for t in range(3):
        if t not in mix_dist.columns:
            mix_dist[t] = 0.0
    mix_dist = mix_dist[[0, 1, 2]]
    global_mix = mix_tab['t'].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0)
    g = float(y_all[tr].mean())
    ptab = pd.DataFrame({'p': pid_all[tr], 't': ptype[tr], 'y': y_all[tr]})
    p_rate = ptab.groupby(['p', 't'])['y'].agg(['sum', 'count'])
    K_SH = 60.0
    p_rate['rate'] = (p_rate['sum'] + K_SH * g) / (p_rate['count'] + K_SH)
    rate_wide = p_rate['rate'].unstack()
    for t in range(3):
        if t not in rate_wide.columns:
            rate_wide[t] = g
    rate_wide = rate_wide[[0, 1, 2]].fillna(g)
    cs_va = cs_all[va_mask]; pid_va = pid_all[va_mask]
    mix_row = mix_dist.reindex(cs_va).fillna(global_mix).to_numpy(np.float64)
    rate_row = rate_wide.reindex(pid_va).fillna(g).to_numpy(np.float64)
    return (mix_row * rate_row).sum(axis=1)


def cluster_bootstrap_rho(d, resid, pid_va, n_boot=300, seed=0):
    uniq = np.unique(pid_va)
    rng = np.random.RandomState(seed)
    idx_by_p = {p: np.flatnonzero(pid_va == p) for p in uniq}
    rhos = []
    for _ in range(n_boot):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_p[p] for p in samp])
        dd = d[idx] - d[idx].mean()
        rr = resid[idx] - resid[idx].mean()
        denom = np.sqrt(np.mean(dd**2) * np.mean(rr**2))
        rhos.append(np.mean(dd*rr) / denom if denom > 1e-14 else 0.0)
    return np.array(rhos)


for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    va = season == vs
    yv = y_all[va]
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    feat = build_feature(upto, va)
    d = feat

    pid_va = pid_all[va]
    boot = cluster_bootstrap_rho(d, resid, pid_va, n_boot=300, seed=1)
    rho_point = boot.mean()
    se_boot = boot.std()
    print(f'\n=== fold {tag} ({vs}) ===')
    print(f'  n={va.sum():,}  투수수={len(np.unique(pid_va))}')
    print(f'  rho(부트스트랩 평균) = {rho_point:+.5f}   SE(부트) = {se_boot:.5f}')
    print(f'  z = {rho_point/se_boot:+.2f}')
    print(f'  95% 부트 CI = [{np.percentile(boot,2.5):+.5f}, {np.percentile(boot,97.5):+.5f}]')
