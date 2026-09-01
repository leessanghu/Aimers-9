"""재학습 없이 캐시만으로: mc6와 겹치는 4헤드(midother/condball/countresid/future50)에서
mc6와 안 겹치는 3헤드(base/hurdle/ordinal)로 가중치를 소폭 이전하면 v117 구조가 개선되는가.

[경고] 이건 '가중치 재배분'이고 오늘 v99/v98/mc5스윕에서 이미 로컬-실측 역전을 여러 번
확인한 카테고리다([[v95-is-local-optimum-confirmed]], [[donor-heads-mattered-real]]).
여기서 나오는 결과는 '실측 프로브를 태울 가치가 있는 가설'로만 취급하고,
로컬 숫자 자체를 확정된 사실로 쓰지 않는다.

방법: 8헤드 비율 안에서 이전비율 t만큼 겹치는4헤드 가중치의 t%를 안겹치는3헤드로 이동.
mc6(0.48)/strk(0.10)는 고정. H1/H2 클린검증(중심화+무절편) + 대조군.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
OVERLAP = ['midother', 'condball', 'countresid', 'future50']
INDEP = ['base', 'hurdle', 'ordinal']
S_MC6, S_STRK = 0.48, 0.10


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


def make_blend(H, W8, tag):
    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    base8 = sum(W8[k] * H[k] for k in HEADS8)
    return np.clip(rest * base8 + S_MC6 * p_mc6 + S_STRK * p_strk, 0, 1)


W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
t0 = sum(W0.values()); W0 = {k: v / t0 for k, v in W0.items()}
overlap_sum = sum(W0[k] for k in OVERLAP)
indep_sum = sum(W0[k] for k in INDEP)

for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    H = build8(tag)
    Xv = X.loc[va]
    mth = Xv['game_month'].to_numpy()
    H1 = mth <= 6; H2 = ~H1
    sc = lambda pp, msk: 1e5 * (1 - np.mean((np.clip(pp[msk], 0, 1) - yv[msk]) ** 2) / B_)

    blend0 = make_blend(H, W0, tag)
    print(f'\n=== fold {tag} ({vs}) : 전체BSS={sc(blend0, np.ones(len(yv),bool)):.1f} ===')
    print(f'{"이전비율t":>10}{"H1->H2":>10}{"H2->H1":>10}{"평균":>10}')
    for t in (0.0, 0.10, 0.20, 0.30, 0.50):
        Wt = dict(W0)
        move = overlap_sum * t
        for k in OVERLAP:
            Wt[k] = W0[k] * (1 - t)
        for k in INDEP:
            Wt[k] = W0[k] + move * (W0[k] / indep_sum)
        blend_t = make_blend(H, Wt, tag)
        gains = []
        d = blend_t - blend0
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            gains.append(sc(blend_t, ev_m) - sc(blend0, ev_m))
        print(f'{t:>10.2f}{gains[0]:>+10.2f}{gains[1]:>+10.2f}{np.mean(gains):>+10.2f}')

    # 대조군: 랜덤 방향으로 같은 크기 이동
    rng = np.random.RandomState(3)
    print(f'  --- 대조군(무작위 재배분, 같은 크기) ---')
    for _ in range(3):
        Wr = dict(W0)
        noise = rng.normal(0, 0.02, len(HEADS8))
        noise -= noise.mean()
        for k, nz in zip(HEADS8, noise):
            Wr[k] = max(0.001, W0[k] + nz)
        s_ = sum(Wr.values()); Wr = {k: v / s_ for k, v in Wr.items()}
        blend_r = make_blend(H, Wr, tag)
        gains = []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            gains.append(sc(blend_r, ev_m) - sc(blend0, ev_m))
        print(f'  random  H1->H2={gains[0]:+7.2f}  H2->H1={gains[1]:+7.2f}  평균={np.mean(gains):+7.2f}')
