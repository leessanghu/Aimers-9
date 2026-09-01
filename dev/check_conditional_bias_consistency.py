"""조건부 레벨 편차가 fold 간에 재현되는가.
전역편차를 뺀 '상대 편차'가 두 fold에서 같은 부호/크기면 구조적 미스캘리브레이션이고,
fold마다 다르면 그 fold 고유 잡음이다."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

K = 1e5 / 0.249807
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


AXES = ['game_type', 'same_hand', 'batter_team_id', 'pitcher_team_id', 'asof_batter_n_q']
store = {}
for tag, vs in [('A', 2024), ('C', 2022), ('B', 2023)]:
    va = season == vs
    yv = y[va]
    try:
        H = build8(tag)
    except Exception:
        print(f'fold {tag}: 캐시 부족, 건너뜀')
        continue
    W = {k: float(v88[f'{k}_weight']) for k in H}
    t = sum(W.values())
    W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = blend - yv
    bg = resid.mean()
    raw = raw_all[raw_all['season'] == vs].reset_index(drop=True)
    Xv = X.loc[va].reset_index(drop=True)
    bn = np.nan_to_num(raw['asof_batter_n'].to_numpy(np.float64), nan=0.0)
    cols = {
        'game_type': raw['game_type'].to_numpy(),
        'same_hand': Xv['same_hand'].to_numpy(),
        'batter_team_id': raw['batter_team_id'].to_numpy(),
        'pitcher_team_id': raw['pitcher_team_id'].to_numpy(),
        'asof_batter_n_q': np.digitize(bn, [1000, 4000, 8000]),
    }
    print(f'\n=== fold {tag} ({vs})  전역편차={bg:+.6f} ===')
    for ax in AXES:
        s = pd.Series(cols[ax])
        rows = []
        for gv, idx in s.groupby(s).groups.items():
            m = np.zeros(len(yv), bool)
            m[np.asarray(idx)] = True
            if m.sum() < 2000:
                continue
            rows.append((gv, m.sum(), resid[m].mean() - bg))
        store.setdefault(ax, {})[tag] = {r[0]: r[2] for r in rows}
        top = sorted(rows, key=lambda r: -abs(r[2]))[:3]
        txt = '  '.join(f'{str(g)[:10]}:{d:+.5f}' for g, n, d in top)
        print(f'  {ax:18s} {txt}')

print('\n\n=== fold 간 상대편차 일치도 (구조적인가?) ===')
for ax, folds in store.items():
    tags = [t for t in ('A', 'C', 'B') if t in folds]
    if len(tags) < 2:
        continue
    keys = set(folds[tags[0]])
    for t in tags[1:]:
        keys &= set(folds[t])
    keys = sorted(keys, key=str)
    if len(keys) < 2:
        continue
    print(f'\n  [{ax}]  공통그룹 {len(keys)}개')
    hdr = '    ' + f'{"그룹":14s}' + ''.join(f'{("fold"+t):>12s}' for t in tags)
    print(hdr)
    for k2 in keys:
        print(f'    {str(k2)[:14]:14s}' + ''.join(f'{folds[t][k2]:>+12.5f}' for t in tags))
    if len(tags) >= 2:
        a = np.array([folds[tags[0]][k2] for k2 in keys])
        b = np.array([folds[tags[1]][k2] for k2 in keys])
        agree = np.mean(np.sign(a) == np.sign(b))
        cc = np.corrcoef(a, b)[0, 1] if len(keys) > 2 else np.nan
        print(f'    -> 부호일치율={agree*100:.0f}%   상관={cc:+.3f}' if len(keys) > 2
              else f'    -> 부호일치율={agree*100:.0f}%')
