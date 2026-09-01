"""idea92 재실행 (leakage 수정판).
이전 버전은 프로덕션 테이블(2019-2024 전체)로 fold A(2024)를 검증해 leakage.
여기서는 fold별로 train<=upto 데이터만으로 슬로프 테이블을 새로 만든다."""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'season', 'pitcher_id', 'strikes_before', 'control_success'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)
df['is_2k'] = (df['strikes_before'] == 2).astype(int)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807

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
allm = np.ones(len(yv), bool)

# *** 핵심: train<=2023 으로만 슬로프 테이블 생성 (leakage 방지) ***
train = df[df.season <= 2023]
g = train.groupby(['pitcher_id', 'is_2k'])['control_success'].agg(['sum', 'count']).unstack(fill_value=0)
n1 = g[('count', 1)] if ('count', 1) in g.columns else pd.Series(0, index=g.index)
s1 = g[('sum', 1)] if ('sum', 1) in g.columns else pd.Series(0, index=g.index)
n0 = g[('count', 0)] if ('count', 0) in g.columns else pd.Series(0, index=g.index)
s0 = g[('sum', 0)] if ('sum', 0) in g.columns else pd.Series(0, index=g.index)
rate1 = s1 / n1.replace(0, np.nan)
rate0 = s0 / n0.replace(0, np.nan)
gap_tbl = (rate1 - rate0)

va_idx = df.index[va]
pid_va = df.loc[va_idx, 'pitcher_id'].to_numpy()
is2k_va = df.loc[va_idx, 'is_2k'].to_numpy().astype(bool)
n2k_va = n1.reindex(pid_va).fillna(0).to_numpy(np.float64)
gap_va = np.nan_to_num(gap_tbl.reindex(pid_va).to_numpy(np.float64), nan=0.0)

print(f'슬로프 테이블: train<=2023 기준, 투수수={len(n1)}')
print(f'v88_final(무보정) = {sc(v88_final, allm):.2f}')
print(f'2K행 비율: {is2k_va.mean()*100:.1f}%')
print()

for K in [1500, 4000]:
    shrunk = gap_va * (n2k_va / (n2k_va + K))
    applied = np.where(is2k_va, shrunk, 0.0)
    print(f'=== K={K} ===')
    for name, mode in [('A 현재(전체행 centered)', 'A'), ('B 2K행만+2K내centered', 'B'), ('C 2K행만+중심화없음', 'C')]:
        gains = []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            if mode == 'A':
                c = applied[fit_m].mean()
                cc = applied - c
                Cv = np.mean(cc[fit_m] * resid[fit_m]); Vv = np.mean(cc[fit_m] ** 2)
                a = Cv / Vv if Vv > 1e-12 else 0.0
                adj = v88_final.copy(); adj[ev_m] = v88_final[ev_m] + a * cc[ev_m]
            elif mode == 'B':
                fit2k = fit_m & is2k_va
                c = applied[fit2k].mean()
                cc = np.where(is2k_va, applied - c, 0.0)
                Cv = np.mean(cc[fit2k] * resid[fit2k]); Vv = np.mean(cc[fit2k] ** 2)
                a = Cv / Vv if Vv > 1e-12 else 0.0
                adj = v88_final.copy()
                m2 = ev_m & is2k_va
                adj[m2] = v88_final[m2] + a * cc[m2]
            else:
                fit2k = fit_m & is2k_va
                cc = applied
                Cv = np.mean(cc[fit2k] * resid[fit2k]); Vv = np.mean(cc[fit2k] ** 2)
                a = Cv / Vv if Vv > 1e-12 else 0.0
                adj = v88_final.copy()
                m2 = ev_m & is2k_va
                adj[m2] = v88_final[m2] + a * cc[m2]
            gains.append(sc(adj, ev_m) - sc(v88_final, ev_m))
        print(f'  {name:26s} H1->H2={gains[0]:+7.2f}  H2->H1={gains[1]:+7.2f}  평균={np.mean(gains):+7.2f}')
    print()
