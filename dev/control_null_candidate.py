"""대조군: 신호가 전혀 없는 후보(순수 노이즈/상수)를 H1/H2 절차에 넣으면 얼마가 나오는가.
여기서 +20 근처가 나오면 오늘의 모든 H1/H2 수치는 절편 복구가 만든 허수다."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
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


rng = np.random.RandomState(0)
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v88[f'{k}_weight']) for k in H}
    t = sum(W.values())
    W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)
    mth = X.loc[va, 'game_month'].to_numpy()
    H1 = mth <= 6
    H2 = ~H1
    print(f'\n=== fold {tag} ({vs})  잔차평균={resid.mean():+.6f}  blend={sc(blend, np.ones(len(yv),bool)):.2f} ===')
    print(f'    H1 잔차평균={resid[H1].mean():+.6f}   H2 잔차평균={resid[H2].mean():+.6f}')

    cands = {
        '순수랜덤노이즈(std .05)': rng.normal(0, 0.05, len(yv)),
        '순수랜덤노이즈(std .01)': rng.normal(0, 0.01, len(yv)),
        '상수 0 (d=0)': np.zeros(len(yv)),
    }
    print('  --- 절편 b 포함 (오늘 쓴 방식) ---')
    for name, d in cands.items():
        gains = []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            mdf, mrf = d[fit_m].mean(), resid[fit_m].mean()
            cv = np.mean((d[fit_m] - mdf) * (resid[fit_m] - mrf))
            vr = np.mean((d[fit_m] - mdf) ** 2)
            a = cv / vr if vr > 1e-14 else 0.0
            b = mrf - a * mdf
            bl = blend.copy()
            bl[ev_m] = blend[ev_m] + a * d[ev_m] + b
            gains.append(sc(bl, ev_m) - sc(blend, ev_m))
        print(f'  {name:26s} H1->H2={gains[0]:+8.2f}  H2->H1={gains[1]:+8.2f}  평균={np.mean(gains):+8.2f}')

    print('  --- 절편 없이 b=0 (순수 방향성만) ---')
    for name, d in cands.items():
        gains = []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            mdf, mrf = d[fit_m].mean(), resid[fit_m].mean()
            cv = np.mean((d[fit_m] - mdf) * (resid[fit_m] - mrf))
            vr = np.mean((d[fit_m] - mdf) ** 2)
            a = cv / vr if vr > 1e-14 else 0.0
            bl = blend.copy()
            bl[ev_m] = blend[ev_m] + a * (d[ev_m] - mdf)
            gains.append(sc(bl, ev_m) - sc(blend, ev_m))
        print(f'  {name:26s} H1->H2={gains[0]:+8.2f}  H2->H1={gains[1]:+8.2f}  평균={np.mean(gains):+8.2f}')
