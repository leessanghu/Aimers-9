"""팀원 피처 중 우리한테 없는 것들을 팀원 pipeline.py로 만들어서
v88_final 잔차에 대해 정직 스크리닝 (H1<->H2, 전역레벨 제외 순수기여).
train<=2023 테이블로 생성 (leakage 방지)."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'dev/teammate_v1')
import numpy as np, pandas as pd, joblib
import pipeline as P
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
unc = 0.249807

tr = season <= 2023
train_df = df.loc[tr].reset_index(drop=True)
tm = P._load_tm()
tables = P.build_tables(train_df)
df_fe = P.apply_fe(df, tm, tables)
log(f'팀원 FE 완료 {df_fe.shape}')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
p = 'A'
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
P11 = np.load('dev/idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
raw = sum(W[k] * H[k] for k in H)
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
v88_final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
resid = yv - v88_final
H1 = mth <= 6; H2 = ~H1

CANDS = ['inseason_K20', 'inseason_K100', 'inseason_K200', 'season_progress',
         'month_x_inseason_n', 'pitcher_ahead', 'pitcher_behind', 'risp_and_two_out',
         'runner_pressure', 'we_gap', 'asof_pitcher_ball_strike_ratio',
         'asof_pitcher_success_shrunk', 'asof_pitcher_reliable',
         'pitcher_success_momentum', 'pitcher_middle_momentum', 'pitcher_recent_vs_career',
         'pitchmix_fb_vs_offspeed', 'pitchmix_breaking_share', 'pitchmix_entropy',
         'matchup_success_diff', 'matchup_middle_diff', 'is_two_strike', 'is_three_ball',
         'is_full_count', 'count_sum', 'tm_speed_drop']
# 파생: K20 vs K200 격차 (multi-K 핵심 아이디어)
df_fe['inseason_K20_minus_K200'] = df_fe['inseason_K20'] - df_fe['inseason_K200']
CANDS.append('inseason_K20_minus_K200')

df_fe_va = df_fe.loc[va].reset_index(drop=True)

print(f'{"feature":32s} {"corr(y)":>9s} {"H1->H2":>9s} {"H2->H1":>9s} {"평균":>8s} {"판정":>6s}')
print('-' * 78)
results = []
for c in CANDS:
    if c not in df_fe_va.columns:
        print(f'{c:32s} (컬럼 없음, 스킵)')
        continue
    v = df_fe_va[c].to_numpy(np.float64)
    v = np.nan_to_num(v, nan=np.nanmedian(v) if np.isfinite(np.nanmedian(v)) else 0.0)
    r = np.corrcoef(v, yv)[0, 1] if v.std() > 1e-12 else 0.0
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        uniq = np.unique(v)
        if len(uniq) < 2:
            gains.append(0.0); continue
        if len(uniq) <= 20:
            edges = np.r_[uniq - 1e-9, uniq[-1] + 1e-9]
        else:
            edges = np.unique(np.quantile(v[fit_m], np.linspace(0, 1, 9)))
            if len(edges) < 3:
                gains.append(0.0); continue
            edges = edges.astype(float); edges[0] -= 1e-9; edges[-1] += 1e-9
        bf_ = np.clip(np.digitize(v[fit_m], edges) - 1, 0, len(edges) - 2)
        be = np.clip(np.digitize(v[ev_m], edges) - 1, 0, len(edges) - 2)
        nbin = len(edges) - 1
        rf = resid[fit_m]; gl = rf.mean()
        cmap = np.zeros(nbin)
        for b in range(nbin):
            m = bf_ == b
            if m.sum() >= 500:
                cmap[b] = rf[m].mean() - gl
        adj = v88_final.copy(); adj[ev_m] = v88_final[ev_m] + cmap[be]
        gains.append(sc(adj, ev_m) - sc(v88_final, ev_m))
    avgg = float(np.mean(gains))
    verdict = 'OK' if min(gains) > 0.3 else ''
    results.append((c, gains[0], gains[1], avgg, verdict))
    print(f'{c:32s} {r:+9.4f} {gains[0]:+9.2f} {gains[1]:+9.2f} {avgg:+8.2f} {verdict:>6s}')

print()
ok = [r for r in results if r[4] == 'OK']
print(f'양방향 모두 +0.3 초과: {len(ok)}개')
for r in sorted(ok, key=lambda t: -t[3]):
    print(f'   {r[0]:32s} 평균 {r[3]:+.2f}')
