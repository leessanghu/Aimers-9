"""pitcher x 상황 random slope 전수 스윕.
2K슬로프가 유일하게 실측 생존(+0.83)한 축이므로, 같은 계열을 체계적으로 훑는다.
각 축: 투수별 (조건=1일때 성공률) - (조건=0일때 성공률), 베이지안 축소 후 조건행에만 적용.
Rule4 안전: train<=2023 통계로만 테이블 구성, 각 행은 자기 pitcher_id + 자기 조건만 참조.
fold A H1<->H2 양방향 + fold C 이중검증.
"""
import numpy as np, pandas as pd, joblib, sys, time
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'season', 'pitcher_id', 'batter_id', 'balls_before',
                          'strikes_before', 'outs_before', 'inning', 'num_runners_on',
                          'batter_hand', 'pitcher_hand', 'li', 'score_diff_pitcher_team',
                          'game_type', 'control_success'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)

# 조건 정의 (전부 각 행 자기 컬럼만으로 판정 가능)
COND = {
    'is_2strike': (df.strikes_before == 2),
    'is_3ball': (df.balls_before == 3),
    'is_0_0': (df.balls_before == 0) & (df.strikes_before == 0),
    'ahead': (df.strikes_before > df.balls_before),
    'behind': (df.balls_before > df.strikes_before),
    'risp': (df.num_runners_on >= 1) & (df.outs_before <= 1),
    'runners_on': (df.num_runners_on >= 1),
    'two_outs': (df.outs_before == 2),
    'late_inning': (df.inning >= 7),
    'first_inning': (df.inning == 1),
    'high_li': (df.li >= df.li.median()),
    'same_hand': (df.pitcher_hand == df.batter_hand),
    'losing': (df.score_diff_pitcher_team < 0),
    'F_league': (df.game_type == 'F'),
}

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
unc = 0.249807
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def build_pred(p):
    H = dict(
        base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{p}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{p}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )
    if p == 'A':
        P11 = np.load('dev/idea75_cache/A_proba11.npy')
        H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
        ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
        H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
        raw = sum(W[k] * H[k] for k in H)
        risk = P11[:, [9, 10]].sum(axis=1)
        cut = np.maximum(0.0, risk - float(v88['risk_thr']))
        return np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)
    keys8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50']
    w8 = {k: W[k] for k in keys8}; t = sum(w8.values()); w8 = {k: v / t for k, v in w8.items()}
    return sum(w8[k] * H[k] for k in keys8)


def eval_slope(name, cond_series, fold, train_upto, val_year, K=1500):
    va = season == val_year
    yv = y[va]
    mth = X.loc[va, 'game_month'].to_numpy()
    pred = build_pred(fold)
    sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
    resid = yv - pred
    H1 = mth <= 6; H2 = ~H1

    c = cond_series.astype(int)
    tr_mask = df.season <= train_upto
    tmp = pd.DataFrame({'pid': df.pitcher_id, 'c': c, 'y': df.control_success})[tr_mask.to_numpy()]
    g = tmp.groupby(['pid', 'c'])['y'].agg(['sum', 'count']).unstack(fill_value=0)
    if ('count', 1) not in g.columns or ('count', 0) not in g.columns:
        return None
    n1 = g[('count', 1)]; s1 = g[('sum', 1)]
    n0 = g[('count', 0)]; s0 = g[('sum', 0)]
    rate1 = s1 / n1.replace(0, np.nan)
    rate0 = s0 / n0.replace(0, np.nan)
    gap = (rate1 - rate0)

    va_idx = df.index[va]
    pid_va = df.loc[va_idx, 'pitcher_id'].to_numpy()
    c_va = c.to_numpy()[va]
    n1_va = n1.reindex(pid_va).fillna(0).to_numpy(np.float64)
    gap_va = np.nan_to_num(gap.reindex(pid_va).to_numpy(np.float64), nan=0.0)
    applied = np.where(c_va == 1, gap_va * (n1_va / (n1_va + K)), 0.0)
    if applied.std() < 1e-9:
        return None
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        cc = applied - applied[fit_m].mean()
        C = np.mean(cc[fit_m] * resid[fit_m]); V = np.mean(cc[fit_m] ** 2)
        a = C / V if V > 1e-12 else 0.0
        adj = pred.copy(); adj[ev_m] = pred[ev_m] + a * cc[ev_m]
        gains.append(sc(adj, ev_m) - sc(pred, ev_m))
    return gains, c_va.mean()


print(f'{"조건축":16s} {"적용비율":>8s} | {"foldA H1->H2":>12s} {"foldA H2->H1":>12s} {"A평균":>8s} | {"foldC평균":>10s} {"판정":>6s}')
print('-' * 92)
rows = []
for name, cond in COND.items():
    rA = eval_slope(name, cond, 'A', 2023, 2024)
    rC = eval_slope(name, cond, 'C', 2021, 2022)
    if rA is None or rC is None:
        print(f'{name:16s} (스킵)')
        continue
    gA, ratio = rA
    gC, _ = rC
    avgA, avgC = np.mean(gA), np.mean(gC)
    ok = 'OK' if (min(gA) > 0.3 and avgC > 0) else ''
    rows.append((name, avgA, avgC, ok))
    print(f'{name:16s} {ratio*100:7.1f}% | {gA[0]:+12.2f} {gA[1]:+12.2f} {avgA:+8.2f} | {avgC:+10.2f} {ok:>6s}')

print()
ok = [r for r in rows if r[3] == 'OK']
print(f'fold A 양방향 양수 + fold C 양수: {len(ok)}개')
for r in sorted(ok, key=lambda t: -t[1]):
    print(f'   {r[0]:16s} foldA={r[1]:+.2f}  foldC={r[2]:+.2f}')
print(f'({time.time()-t0:.0f}s)')
