"""SVD 잠재요인: 투수x타자팀 상호작용 (새로운 정규화 원리).
MLP+임베딩(idea61-69)은 개별 파라미터 자유학습이라 노이즈를 외웠다(feature dilution).
SVD는 저차원(rank-k) 강제 압축이라 다른 정규화 - 대량의 셀에 걸쳐 공유되는
'저차원 스타일 매치' 구조만 남기고 나머지는 자동으로 버려진다.

Rule4 안전: 각 행은 자기 pitcher_id x batter_team_id만 참조, train전체 통계로
사전에 fit한 저차원 팩터 테이블을 lookup. test 행간 참조 없음.
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'season', 'pitcher_id', 'batter_team_id', 'control_success'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)

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


def build_v88_final(p):
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
    else:
        keys8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50']
        w8 = {k: W[k] for k in keys8}; t = sum(w8.values()); w8 = {k: v / t for k, v in w8.items()}
        return sum(w8[k] * H[k] for k in keys8)


def fit_svd_table(train, k, shrink_k=50.0):
    """(pitcher, team) 셀 residual(투수자기평균 대비) 을 베이지안축소 후 rank-k SVD."""
    pmean = train.groupby('pitcher_id')['control_success'].transform('mean')
    train = train.assign(resid=train['control_success'] - pmean)
    cell = train.groupby(['pitcher_id', 'batter_team_id'])['resid'].agg(['sum', 'count'])
    shrunk = cell['sum'] / (cell['count'] + shrink_k)  # 베이지안식 축소 (분모에 K 더함, 분자는 이미 0-중심)
    mat = shrunk.unstack(fill_value=0.0)  # (pitcher x team)
    pid_index = mat.index.to_numpy()
    team_index = mat.columns.to_numpy()
    M = mat.to_numpy()
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    k = min(k, len(S))
    M_lowrank = (U[:, :k] * S[:k]) @ Vt[:k, :]
    return pid_index, team_index, M_lowrank


def lookup(pid_index, team_index, M_lowrank, pid_arr, team_arr):
    pid_pos = {v: i for i, v in enumerate(pid_index)}
    team_pos = {v: i for i, v in enumerate(team_index)}
    out = np.zeros(len(pid_arr))
    for i, (p, t) in enumerate(zip(pid_arr, team_arr)):
        ip = pid_pos.get(p); it = team_pos.get(t)
        if ip is not None and it is not None:
            out[i] = M_lowrank[ip, it]
    return out


def run_fold(p, train_upto, val_year):
    va = season == val_year
    yv = y[va]
    mth = X.loc[va, 'game_month'].to_numpy()
    v88_final = build_v88_final(p)
    sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
    resid_base = yv - v88_final
    H1 = mth <= 6; H2 = ~H1

    train = df[df.season <= train_upto]
    va_idx = df.index[va]
    pid_va = df.loc[va_idx, 'pitcher_id'].to_numpy()
    team_va = df.loc[va_idx, 'batter_team_id'].to_numpy()

    print(f'--- fold {p} (train<={train_upto} -> val {val_year}) ---')
    for k in [1, 2, 3, 5, 8, 15]:
        pid_idx, team_idx, M_lr = fit_svd_table(train, k)
        sig = lookup(pid_idx, team_idx, M_lr, pid_va, team_va)
        gains = []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            center = sig[fit_m].mean()
            cc = sig - center
            C = np.mean(cc[fit_m] * resid_base[fit_m])
            V = np.mean(cc[fit_m] ** 2)
            a = C / V if V > 1e-12 else 0.0
            adj = v88_final.copy(); adj[ev_m] = v88_final[ev_m] + a * cc[ev_m]
            gains.append(sc(adj, ev_m) - sc(v88_final, ev_m))
        print(f'  rank k={k:3d}  sig.std={sig.std():.5f}  H1->H2={gains[0]:+7.2f}  H2->H1={gains[1]:+7.2f}  평균={np.mean(gains):+7.2f}')
    print()


run_fold('A', 2023, 2024)
run_fold('C', 2021, 2022)
