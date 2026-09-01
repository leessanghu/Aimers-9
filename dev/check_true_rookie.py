"""asof_pitcher_n == 0 (진짜 완전 첫 투구, 시즌내 누적조차 0)만 따로 분석.
fold A/C 양쪽에서 own-variance BSS/corr/편차 확인."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


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


for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v88[f'{k}_weight']) for k in H}
    t = sum(W.values())
    W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    raw = raw_all[raw_all['season'] == vs].reset_index(drop=True)
    n_ = np.nan_to_num(raw['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)

    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    for thr, lbl in [(0, 'n==0(완전첫투구)'), (5, 'n<5'), (30, 'n<30'), (208, 'n<하위10%')]:
        m = n_ < (thr + 0.5) if thr == 0 else n_ < thr
        if m.sum() < 30:
            print(f'  {lbl:16s} n={m.sum()} (표본부족)')
            continue
        yy, pp = yv[m], blend[m]
        r = yy.mean()
        var_own = max(r * (1 - r), 1e-6)
        bs = np.mean((pp - yy) ** 2)
        bss_own = 1e5 * (1 - bs / var_own)
        corr = np.corrcoef(pp, yy)[0, 1] if pp.std() > 0 and m.sum() > 2 else np.nan
        bias = pp.mean() - yy.mean()
        print(f'  {lbl:16s} n={m.sum():>6,}  실제={r:.4f}  예측={pp.mean():.4f}  편차={bias:+.5f}  '
              f'자체BSS={bss_own:7.1f}  corr={corr:+.4f}')
    # 전체 대비
    yy, pp = yv, blend
    r = yy.mean(); var_own = r*(1-r)
    bs = np.mean((pp-yy)**2)
    print(f'  {"전체(참고)":16s} n={len(yv):>6,}  실제={r:.4f}  예측={pp.mean():.4f}  편차={pp.mean()-yy.mean():+.5f}  '
          f'자체BSS={1e5*(1-bs/var_own):7.1f}  corr={np.corrcoef(pp,yy)[0,1]:+.4f}')
