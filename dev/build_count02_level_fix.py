"""0-2카운트(count_state==2) 모집단 수준 레벨보정. 투수개인차 아니라 그냥 그 카운트
전체에 상수 하나. LGBM이 이 구간에서 편차 거의 0을 보인 것에서 착안.
3종 세트: 대조군 / 중심화+무절편 / fold A+C."""
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
    cs = (raw['balls_before'] * 4 + raw['strikes_before']).to_numpy()
    is02 = (cs == 2).astype(np.float64)

    resid = yv - blend
    mth = raw['game_month'].to_numpy()
    H1 = mth <= 6; H2 = ~H1
    sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / 0.249807)
    rng = np.random.RandomState(7)

    def run(dd):
        gains, coefs = [], []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            mdf = dd[fit_m].mean(); mrf = resid[fit_m].mean()
            cv = np.mean((dd[fit_m]-mdf)*(resid[fit_m]-mrf))
            vr = np.mean((dd[fit_m]-mdf)**2)
            a = cv/vr if vr > 1e-14 else 0.0
            bl = blend.copy()
            bl[ev_m] = blend[ev_m] + a*(dd[ev_m]-mdf)
            gains.append(sc(bl, ev_m) - sc(blend, ev_m))
            coefs.append(a)
        return gains, coefs

    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    n02 = is02.sum()
    print(f'  0-2카운트 행 = {n02:,} ({n02/len(yv)*100:.1f}%)')
    ctrl = rng.normal(0, is02.std(), len(yv))
    gc, _ = run(ctrl)
    print(f'  대조군       평균={np.mean(gc):+7.2f}')
    g, c = run(is02 - is02.mean())
    print(f'  0-2레벨보정  H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}  a={c[0]:+.5f}/{c[1]:+.5f}')
