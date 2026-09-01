"""fold A(train<=2023 -> 2024)에서 train에 단 한 번도 없던 pitcher_id가 얼마나 되고,
그 행들에서 우리 모델이 어떻게 예측하는지(레벨편차, 판별력) 확인.
own-variance BSS로 아티팩트 없이 재고, fold C로 재현성까지 확인."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
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
    train_pids = set(raw_all.loc[raw_all['season'] <= upto, 'pitcher_id'].unique())
    unseen = ~raw['pitcher_id'].isin(train_pids).to_numpy()

    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  전체 검증행 = {len(yv):,}')
    print(f'  train에 없던 신규 pitcher_id 행 = {unseen.sum():,} ({unseen.mean()*100:.2f}%)')
    print(f'  신규 투수 수 = {raw.loc[unseen, "pitcher_id"].nunique()}명')

    for name, m in [('신규(unseen)', unseen), ('기존(seen)', ~unseen)]:
        yy, pp = yv[m], blend[m]
        r = yy.mean()
        var_own = max(r * (1 - r), 1e-6)
        bs = np.mean((pp - yy) ** 2)
        bss_own = 1e5 * (1 - bs / var_own)
        bss_fix = 1e5 * (1 - bs / 0.249807)
        corr = np.corrcoef(pp, yy)[0, 1] if pp.std() > 0 else np.nan
        bias = pp.mean() - yy.mean()
        print(f'    {name:14s} n={m.sum():>7,}  실제성공률={r:.4f}  예측평균={pp.mean():.4f}  '
              f'편차={bias:+.5f}  고정BSS={bss_fix:7.1f}  자체BSS={bss_own:7.1f}  corr={corr:+.4f}')

    # asof_pitcher_n 자체 분포 (신규는 몇으로 채워지나)
    n_ = np.nan_to_num(raw['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
    print(f'    신규투수 asof_pitcher_n: min={n_[unseen].min():.0f} max={n_[unseen].max():.0f} '
          f'평균={n_[unseen].mean():.0f}  (0이 아니면 다른시즌 이력은 있다는 뜻)')
    print(f'    기존투수 asof_pitcher_n 평균={n_[~unseen].mean():.0f}')
