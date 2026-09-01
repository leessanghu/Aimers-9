"""진짜 판별력 약점 스캔. 레벨편차(scan_conditional_bias)와 다른 질문:
'이 그룹에서 우리 모델이 그 그룹 자체 분산 대비 얼마나 잘 판별하는가'
BSS_own = 1e5*(1 - BS/그룹자체분산). 이건 레벨편차 문제를 자동으로 어느정도 포함하지만
핵심은 corr(p,y)로 순수 판별력만 따로 본다.
K그룹 우연기준선 적용 + fold C 재현까지 한 번에."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

K_CONST = 1e5 / 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


def get_fold(tag, vs):
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v88[f'{k}_weight']) for k in H}
    t = sum(W.values())
    W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    raw = raw_all[raw_all['season'] == vs].reset_index(drop=True)
    Xv = X.loc[va].reset_index(drop=True)
    return yv, blend, raw, Xv


def axis_cols(raw, Xv):
    bn = np.nan_to_num(raw['asof_batter_n'].to_numpy(np.float64), nan=0.0)
    pn = np.nan_to_num(raw['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
    return {
        'game_type': raw['game_type'].to_numpy(),
        'same_hand': Xv['same_hand'].to_numpy(),
        'count_state': Xv['count_state'].to_numpy(),
        'outs_before': raw['outs_before'].to_numpy(),
        'num_runners_on': raw['num_runners_on'].to_numpy(),
        'inning_q': np.clip(raw['inning'].to_numpy(), 1, 9),
        'batter_n_q': np.digitize(bn, [1000, 4000, 8000]),
        'pitcher_n_q': np.digitize(pn, [629, 1618, 3355]),
        'game_month': raw['game_month'].to_numpy(),
        'top_bottom': raw['top_bottom'].to_numpy(),
        'li_q': np.digitize(np.nan_to_num(raw['li'].to_numpy(np.float64), nan=0.78), [0.3, 0.7, 1.3]),
        'score_diff_pitcher_team_q': np.digitize(raw['score_diff_pitcher_team'].to_numpy(), [-2, 0, 2]),
    }


def scan(yv, blend, groups, min_n=2000):
    s = pd.Series(groups)
    rows = []
    for gv, idx in s.groupby(s).groups.items():
        m = np.zeros(len(yv), bool)
        m[np.asarray(idx)] = True
        if m.sum() < min_n:
            continue
        yy, pp = yv[m], blend[m]
        r = yy.mean()
        var_own = max(r * (1 - r), 1e-6)
        bs = np.mean((pp - yy) ** 2)
        bss_own = 1e5 * (1 - bs / var_own)
        corr = np.corrcoef(pp, yy)[0, 1] if pp.std() > 0 and yy.std() > 0 else np.nan
        rows.append((gv, m.sum(), bss_own, corr, var_own))
    return rows


print('=== fold A(2024) 축별 판별력(corr) 최소~최대 편차 ===')
yv_a, blend_a, raw_a, Xv_a = get_fold('A', 2024)
cols_a = axis_cols(raw_a, Xv_a)
n_a = len(yv_a)
result_a = {}
for ax, g in cols_a.items():
    rows = scan(yv_a, blend_a, g)
    if len(rows) < 2:
        continue
    corrs = [r[3] for r in rows if np.isfinite(r[3])]
    if not corrs:
        continue
    K_grp = len(rows)
    span = max(corrs) - min(corrs)
    result_a[ax] = rows
    worst = min(rows, key=lambda r: r[3] if np.isfinite(r[3]) else 1)
    best = max(rows, key=lambda r: r[3] if np.isfinite(r[3]) else -1)
    print(f'  {ax:26s} K={K_grp:2d}  corr범위=[{min(corrs):+.4f}, {max(corrs):+.4f}]  폭={span:.4f}  '
          f'최약={str(worst[0])[:10]}(n={worst[1]:,},corr={worst[3]:+.4f})')

print()
print('=== fold C(2022) 재현 확인 (corr 폭 상위 축만) ===')
yv_c, blend_c, raw_c, Xv_c = get_fold('C', 2022)
cols_c = axis_cols(raw_c, Xv_c)
top_axes = sorted(result_a.keys(), key=lambda a: -(max(r[3] for r in result_a[a] if np.isfinite(r[3])) - min(r[3] for r in result_a[a] if np.isfinite(r[3]))))[:5]
for ax in top_axes:
    rows_c = scan(yv_c, blend_c, cols_c[ax])
    da = {r[0]: r[3] for r in result_a[ax]}
    dc = {r[0]: r[3] for r in rows_c}
    common = sorted(set(da) & set(dc), key=str)
    if len(common) < 2:
        continue
    a_ = np.array([da[k2] for k2 in common])
    c_ = np.array([dc[k2] for k2 in common])
    valid = np.isfinite(a_) & np.isfinite(c_)
    if valid.sum() < 2:
        continue
    cc = np.corrcoef(a_[valid], c_[valid])[0, 1] if valid.sum() > 2 else np.nan
    print(f'\n  [{ax}]  공통그룹={valid.sum()}')
    for k2 in common:
        if k2 in da and k2 in dc:
            print(f'    {str(k2):12s} foldA_corr={da[k2]:+.4f}  foldC_corr={dc[k2]:+.4f}')
    print(f'    -> fold간 상관 = {cc:+.3f}')
