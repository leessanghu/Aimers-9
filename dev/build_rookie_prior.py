"""신인전용 사전확률 보정. 전역평균 대신 '신인(n=0)만의 평균'으로 축소.
d = rookie_prior - global_mean, 가중치는 asof_pitcher_n이 작을수록 크게(지수감쇠).
3종 검증: 대조군 / 중심화+무절편 / fold A+C."""
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
    n_ = np.nan_to_num(raw['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)

    # train(<=upto)에서 신인평균(그 시즌 첫 등판, asof_pitcher_n==0)과 전역평균 계산
    tr = raw_all[raw_all['season'] <= upto]
    tr_n = np.nan_to_num(tr['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
    tr_y = tr['control_success'].to_numpy(np.float64)
    rookie_mean = tr_y[tr_n < 0.5].mean()
    global_mean = tr_y.mean()
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  train 신인(n=0)평균={rookie_mean:.4f}  전역평균={global_mean:.4f}  차이={rookie_mean-global_mean:+.4f}')

    HALF_LIFE = 100.0  # n이 이 정도 쌓이면 영향력 절반
    w_rookie = 2.0 ** (-n_ / HALF_LIFE)
    d = w_rookie * (rookie_mean - global_mean)
    d = d - d.mean()

    resid = yv - blend
    mth = raw['game_month'].to_numpy()
    H1 = mth <= 6
    H2 = ~H1
    sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / 0.249807)
    rng = np.random.RandomState(3)
    ctrl = rng.normal(0, d.std(), len(yv))

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

    gc, _ = run(ctrl)
    gr, cf = run(d)
    print(f'  대조군       평균={np.mean(gc):+7.2f}')
    print(f'  신인prior축소 H1->H2={gr[0]:+7.2f}  H2->H1={gr[1]:+7.2f}  평균={np.mean(gr):+7.2f}  a={cf[0]:+.3f}/{cf[1]:+.3f}')

    for half in (30.0, 300.0, 1000.0):
        w2 = 2.0 ** (-n_ / half)
        d2 = w2 * (rookie_mean - global_mean); d2 = d2 - d2.mean()
        g2, c2 = run(d2)
        print(f'    half_life={half:6.0f}  H1->H2={g2[0]:+7.2f}  H2->H1={g2[1]:+7.2f}  평균={np.mean(g2):+7.2f}')
