"""저경험 투수 x 첫이닝(1회) 모집단 수준 상호작용. asof_pitcher_n이 낮은 투수가
1회에 유독 못하는가? 개인편차가 아니라 그룹 전체 효과로 검증.
3종 세트: 대조군 / 중심화+무절편 / fold A+C 재현."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

K_CONST = 1e5 / 0.249807
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


def honest_test(yv, blend, d, mth, tag_lbl):
    resid = blend - yv  # 편차(예측-실제) - v88_final 기준 통일
    resid = yv - blend  # 실제-예측(기존 관례 유지)
    H1 = mth <= 6
    H2 = ~H1
    sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / 0.249807)
    rng = np.random.RandomState(2)
    ctrl = rng.normal(0, d.std() if d.std() > 0 else 0.02, len(yv))

    def run(dd):
        gains, coefs = [], []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            mdf = dd[fit_m].mean()
            mrf = resid[fit_m].mean()
            cv = np.mean((dd[fit_m] - mdf) * (resid[fit_m] - mrf))
            vr = np.mean((dd[fit_m] - mdf) ** 2)
            a = cv / vr if vr > 1e-14 else 0.0
            bl = blend.copy()
            bl[ev_m] = blend[ev_m] + a * (dd[ev_m] - mdf)
            gains.append(sc(bl, ev_m) - sc(blend, ev_m))
            coefs.append(a)
        return gains, coefs

    gc, _ = run(ctrl)
    gr, cf = run(d)
    print(f'  [{tag_lbl}] 대조군 평균={np.mean(gc):+6.2f}   신호 H1->H2={gr[0]:+7.2f} H2->H1={gr[1]:+7.2f} '
          f'평균={np.mean(gr):+7.2f}  a={cf[0]:+.4f}/{cf[1]:+.4f}')
    return np.mean(gr), np.mean(gc)


results = {}
for tag, vs in [('A', 2024), ('C', 2022)]:
    yv, blend, raw, Xv = get_fold(tag, vs)
    n_ = np.nan_to_num(raw['asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
    inning = raw['inning'].to_numpy()
    is_first_pitch_of_game = (Xv['inning_n'].to_numpy() < np.log1p(1)) if 'inning_n' in Xv.columns else None

    print(f'\n=== fold {tag} ({vs}) ===')
    n_tr = np.nan_to_num(raw_all.loc[raw_all['season'] <= (vs - 1), 'asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
    low_thr = np.quantile(n_tr, 0.10)
    print(f'  저경험 임계값(train 10%ile) = {low_thr:.0f}')

    is_low = (n_ < low_thr).astype(np.float64)
    is_inn1 = (inning == 1).astype(np.float64)
    d = is_low * is_inn1
    d = d - d.mean()
    mth = raw['game_month'].to_numpy()
    g, c = honest_test(yv, blend, d, mth, '저경험x1회')
    results.setdefault('저경험x1회', {})[tag] = (g, c, (is_low * is_inn1).sum())

    # 더 좁게: 저경험 x 1회 x 첫타자(카운트 0-0)
    is_first_pa = (Xv['count_state'].to_numpy() == 0).astype(np.float64)
    d2 = is_low * is_inn1 * is_first_pa
    d2 = d2 - d2.mean()
    g2, c2 = honest_test(yv, blend, d2, mth, '저경험x1회x0-0카운트')
    results.setdefault('저경험x1회x0-0', {})[tag] = (g2, c2, (is_low * is_inn1 * is_first_pa).sum())

print('\n=== 종합 ===')
for name, r in results.items():
    if 'A' in r and 'C' in r:
        gA, cA, nA = r['A']
        gC, cC, nC = r['C']
        print(f'  {name:20s} foldA={gA:+.2f}(대조군{cA:+.2f}, n={nA:.0f})  foldC={gC:+.2f}(대조군{cC:+.2f}, n={nC:.0f})')
