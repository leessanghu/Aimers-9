"""기하학자의 메커니즘 주장 직접 검증:
"트리는 대각선(지배방향)을 축정렬 계단으로 재발견하며 분할경계마다 지터를 쌓는다.
 그래서 지터는 지배방향이 가파른 곳에 집중된다. 단일지표 추출은 거기를 표적 제거한다."

검증법: 시드쌍(s42, s7)의 차이 = 순수 지터. 이걸 단일지표 s의 십분위별로 분해해서
  - 지터가 s의 '가파른 구간'(중앙부)에 집중돼 있으면 -> 메커니즘 주장 성립
  - 십분위 전체에 고르게 퍼져 있으면 -> 지터는 등방적이고 표적 제거 대상이 없음
      (그러면 레버A는 배깅 대비 추가이득이 없고, 그냥 배깅이 최적)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.linear_model import Ridge

B = 0.249807
K = 1e5 / B

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

SEED_PAIRS = {
    'multires':   'dev/idea13_cache/{t}_multires_s{s}.npy',
    'ordinal':    'dev/idea13_cache/{t}_ordinal_s{s}.npy',
    'midother':   'dev/idea46_cache/{t}_midother_s{s}.npy',
    'condball':   'dev/idea54_cache/{t}_cond_ball_s{s}.npy',
    'countresid': 'dev/idea54_cache/{t}_count_resid_s{s}.npy',
    'future50':   'dev/idea54_cache/{t}_future50_multi_s{s}.npy',
}


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


for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    Xv = X.loc[va, FEAT].astype(np.float64)
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t = sum(W.values()); W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)

    # 단일지표 s (v95 로짓의 1차 성분)
    lg = np.log(np.clip(blend, 1e-6, 1 - 1e-6) / (1 - np.clip(blend, 1e-6, 1 - 1e-6)))
    Xs = np.nan_to_num(Xv.to_numpy(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    Z = (Xs - Xs.mean(0)) / (Xs.std(0) + 1e-9)
    s = Z @ Ridge(alpha=10.0).fit(Z, lg).coef_

    # 블렌드 수준 지터: 가중합된 시드차이
    jitter = np.zeros(len(yv))
    for head, pat in SEED_PAIRS.items():
        p1 = np.load(pat.format(t=tag, s=42))
        p2 = np.load(pat.format(t=tag, s=7))
        jitter += W[head] * (p1 - p2)

    # s의 십분위별 지터 크기 + 그 구간의 '가파름'(dE[y]/ds)
    qs = np.quantile(s, np.linspace(0, 1, 11)[1:-1])
    dec = np.digitize(s, qs)
    print(f'\n=== fold {tag} ({vs}) : 단일지표 s 십분위별 지터 분해 ===')
    print(f'{"십분위":>6}{"n":>9}{"s평균":>10}{"blend평균":>11}{"가파름|ds|":>12}'
          f'{"지터sd":>10}{"지터분산비중":>13}')
    tot_var = float(np.mean(jitter ** 2))
    rows = []
    for d in range(10):
        m = dec == d
        smean = s[m].mean()
        bmean = blend[m].mean()
        jsd = float(np.std(jitter[m]))
        share = float(np.mean(jitter[m] ** 2)) * m.mean() / tot_var
        rows.append((d, m.sum(), smean, bmean, jsd, share))
    # 가파름: 인접 십분위 blend평균 차이 / s평균 차이
    for i, (d, n_, smean, bmean, jsd, share) in enumerate(rows):
        if i == 0:
            slope = abs((rows[1][3] - rows[0][3]) / (rows[1][2] - rows[0][2] + 1e-12))
        elif i == 9:
            slope = abs((rows[9][3] - rows[8][3]) / (rows[9][2] - rows[8][2] + 1e-12))
        else:
            slope = abs((rows[i+1][3] - rows[i-1][3]) / (rows[i+1][2] - rows[i-1][2] + 1e-12))
        print(f'{d:>6}{n_:>9,}{smean:>10.3f}{bmean:>11.4f}{slope:>12.4f}'
              f'{jsd:>10.5f}{share*100:>12.1f}%')

    shares = np.array([r[5] for r in rows])
    print(f'  균등분포라면 각 10.0%.  최대/최소 비율 = {shares.max()/shares.min():.2f}배')
    print(f'  중앙4분위(3~6) 지터분산 비중 = {shares[3:7].sum()*100:.1f}%  (균등이면 40.0%)')

print('\n[판정] 중앙부 비중이 40%를 크게 넘으면 -> 지배방향 가파른 곳에 지터 집중(메커니즘 성립)')
print('       40% 근처면 -> 지터는 등방적. 단일지표 표적제거로 얻을 추가이득 없음(배깅이 최적)')
