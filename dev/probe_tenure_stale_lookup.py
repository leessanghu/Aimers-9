"""'룩업피처가 신인/베테랑에서 다르게 열화되는가' 검증 (전역 D_true=-0.00097은 상쇄된
반대부호를 못 잡음 -> 구간별 C/V로 직접 측정).

투수를 '해당 시즌 기준 경력 연차'(직전까지 등장한 시즌 수)로 나눠서, v95 블렌드의
잔차(=y-blend)가 경력 구간별로 순수 레벨시프트(그 구간 평균만큼 이동)했을 때
이득이 있는지 측정. 이건 XGB 등 새 모델이 아니라 '그 구간 평균만큼 밀어주는' 가장
단순한 보정이라, 지금까지 실패한 것들(모델 강도/피처/타겟분해)과 완전히 다른 축이다.

H = 구간별 최대이득 / 전역 최대이득. 대조군(무작위 동일크기 분할)과 비교해서
경력축이 진짜 신호인지 판별. fold A/C 둘 다 재현되는지도 확인.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
pid_all = raw_all['pitcher_id'].to_numpy()
season_all = raw_all['season'].to_numpy()
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


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


def tenure_years(vs, pid_va):
    """vs 시즌 기준, 그 이전 시즌들 중 해당 투수가 몇 개 시즌에 등장했는지(경력 연차)."""
    prior = raw_all[season_all < vs]
    seen_seasons = prior.groupby('pitcher_id')['season'].nunique()
    return pd.Series(pid_va).map(seen_seasons).fillna(0).to_numpy(np.int64)


def analyze(d, resid, labels):
    N = len(resid)
    C = float(np.sum(resid * d) / N); V = float(np.sum(d * d) / N)
    g_glob = K * C ** 2 / V if V > 1e-15 else 0.0
    tot = 0.0; sj = {}
    for j in np.unique(labels):
        m = labels == j
        Cj = float(np.sum(resid[m] * d[m]) / N); Vj = float(np.sum(d[m] * d[m]) / N)
        if Vj <= 1e-15:
            continue
        tot += K * Cj ** 2 / Vj
        sj[int(j)] = (-Cj / Vj, int(m.sum()), float(resid[m].mean()))
    return (tot / g_glob if g_glob > 1e-15 else 0.0), g_glob, tot, sj


results = {}
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t = sum(W.values()); W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend

    pid_va = pid_all[va]
    ten = tenure_years(vs, pid_va)
    # 구간: 0=신인(경력0), 1=1~2년차, 2=3~5년차, 3=6년+(베테랑)
    bucket = np.where(ten == 0, 0, np.where(ten <= 2, 1, np.where(ten <= 5, 2, 3)))
    d = np.ones(len(yv))  # 순수 레벨시프트 검증용 dummy(구간별 C/V만 봄, d=1 상수)

    h, g_glob, g_region, sj = analyze(d, resid, bucket)
    # 대조군: 같은 크기분포 무작위 분할 5회 평균
    rng = np.random.RandomState(3)
    hs = []
    for _ in range(5):
        hs.append(analyze(d, resid, rng.permutation(bucket))[0])
    h_ctrl = float(np.mean(hs))

    print(f'\n=== fold {tag} (경력축) ===')
    names = {0: '신인(0)', 1: '1~2년차', 2: '3~5년차', 3: '6년+(베테랑)'}
    for j in sorted(sj):
        s_opt, n, rm = sj[j]
        print(f'  {names[j]:<12} n={n:>7,}  잔차평균={rm:+.5f}  (레벨시프트 s*={s_opt:+.5f})')
    print(f'  구간별 총 이득(경력축) = {g_region:+.2f}점   H={h:.2f}   대조군H={h_ctrl:.2f}')
    results[tag] = sj

print('\n=== fold A/C 재현성: 각 구간의 잔차평균 부호 비교 ===')
names = {0: '신인(0)', 1: '1~2년차', 2: '3~5년차', 3: '6년+(베테랑)'}
for j in sorted(names):
    a = results['A'].get(j); c = results['C'].get(j)
    if a is None or c is None:
        continue
    ok = 'O' if np.sign(a[2]) == np.sign(c[2]) else 'X'
    print(f'  {names[j]:<12} foldA 잔차평균={a[2]:+.5f}   foldC 잔차평균={c[2]:+.5f}   부호일치={ok}')
