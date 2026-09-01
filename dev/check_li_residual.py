"""li(레버리지)의 '게임상황만으로 설명 안 되는 잔차' 성분이 우리 블렌드 잔차와
상관있는가. li 자체는 이미 피처로 쓰이지만, 이 분해된 성분은 명시적으로 준 적 없다.
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
    tr = raw_all[raw_all['season'] <= upto]

    # train으로 (inning,outs,base_state,score_diff_home 클립) -> li 평균 룩업 테이블
    key_tr = list(zip(tr['inning'].clip(upper=10), tr['outs_before'],
                       tr['base_state'], tr['score_diff_home'].clip(-6, 6)))
    li_lookup = pd.Series(tr['li'].to_numpy(), index=pd.Index(key_tr)).groupby(level=0).mean()
    key_va = list(zip(raw['inning'].clip(upper=10), raw['outs_before'],
                       raw['base_state'], raw['score_diff_home'].clip(-6, 6)))
    li_base = pd.Series(key_va).map(li_lookup).to_numpy(np.float64)
    li_base = np.nan_to_num(li_base, nan=tr['li'].mean())
    li_resid = raw['li'].to_numpy(np.float64) - li_base

    resid = yv - blend
    mth = raw['game_month'].to_numpy()
    H1 = mth <= 6
    H2 = ~H1
    sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / 0.249807)
    rng = np.random.RandomState(4)

    def run(dd):
        gains, coefs = [], []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            mdf = dd[fit_m].mean(); mrf = resid[fit_m].mean()
            cv = np.mean((dd[fit_m] - mdf) * (resid[fit_m] - mrf))
            vr = np.mean((dd[fit_m] - mdf) ** 2)
            a = cv / vr if vr > 1e-14 else 0.0
            bl = blend.copy()
            bl[ev_m] = blend[ev_m] + a * (dd[ev_m] - mdf)
            gains.append(sc(bl, ev_m) - sc(blend, ev_m))
            coefs.append(a)
        return gains, coefs

    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  li_resid std = {li_resid.std():.4f}')
    ctrl = rng.normal(0, li_resid.std(), len(yv))
    gc, _ = run(ctrl)
    print(f'  대조군       평균={np.mean(gc):+7.2f}')
    g_raw, c_raw = run(raw['li'].to_numpy(np.float64))
    print(f'  li 원본 그대로     H1->H2={g_raw[0]:+7.2f} H2->H1={g_raw[1]:+7.2f} 평균={np.mean(g_raw):+7.2f} a={c_raw[0]:+.4f}/{c_raw[1]:+.4f}')
    g_res, c_res = run(li_resid)
    print(f'  li_resid(잔차성분)  H1->H2={g_res[0]:+7.2f} H2->H1={g_res[1]:+7.2f} 평균={np.mean(g_res):+7.2f} a={c_res[0]:+.4f}/{c_res[1]:+.4f}')
