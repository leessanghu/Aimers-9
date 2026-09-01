"""콜드스타트(저경험 투수) 보정 설계 + 이중검증(fold A, fold C).
8헤드 버전(base/hurdle/multires/ordinal/midother/condball/countresid/future50)으로
양쪽 폴드에서 동일 아키텍처 사용 (mc5/ingame은 fold C 캐시가 없어서 제외 - apples-to-apples).
해로운5개(multires/midother/condball/countresid/future50)는 head-shrinkage-finding대로 λ=0.2.

절차 (각 fold, 양방향 H1<->H2):
  1) 전역 단일 레벨보정 (fit구간 평균잔차) - 이미 확정된 baseline
  2) 그 위에 경험량(asof_pitcher_n) 구간별 '추가' 보정 - fit구간에서만 추정
  3) 두 폴드 모두에서 이득이 나는지, 특히 저경험 구간에서 나는지 확인
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
unc = 0.249807

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W8 = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
          ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
          countresid=v88['countresid_weight'], future50=v88['future50_weight'])
HARM = ['multires', 'midother', 'condball', 'countresid', 'future50']
W8s = {k: (v * 0.2 if k in HARM else v) for k, v in W8.items()}
t = sum(W8s.values()); W8s = {k: v / t for k, v in W8s.items()}
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def load8(p):
    return dict(
        base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{p}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{p}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


BUCKETS = [(0, 0), (1, 10), (11, 50), (51, 200), (201, 10**9)]


def bucket_id(apn):
    b = np.full(len(apn), -1)
    for i, (lo, hi) in enumerate(BUCKETS):
        b[(apn >= lo) & (apn <= hi)] = i
    return b


results = {}
for p, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    mth = X.loc[va, 'game_month'].to_numpy()
    apn = np.expm1(X.loc[va, 'asof_pitcher_n'].to_numpy()).round().astype(int)
    H = load8(p)
    pred = sum(W8s[k] * H[k] for k in W8s)
    bid = bucket_id(apn)

    sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
    resid = yv - pred
    H1 = mth <= 6; H2 = ~H1

    print(f'=== fold {p} (val {vs}) ===')
    print(f'무보정 전체 BSS = {sc(pred, np.ones(len(yv), bool)):.1f}')
    print('구간별 표본수 및 무보정 편차:')
    for i, (lo, hi) in enumerate(BUCKETS):
        m = bid == i
        if m.sum() == 0:
            continue
        print(f'  n={lo}-{hi if hi<10**8 else "+"}: 행={m.sum():>7,} ({m.mean()*100:.3f}%)  편차={resid[m].mean()*-1:+.4f}  BSS={sc(pred,m):8.1f}')
    print()

    gains_global = []
    gains_bucket = []
    bucket_detail = {i: [] for i in range(len(BUCKETS))}
    for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
        # 1) 전역 보정
        shift = resid[fit_m].mean()
        adj_global = pred.copy(); adj_global[ev_m] = pred[ev_m] + shift
        g_global = sc(adj_global, ev_m) - sc(pred, ev_m)
        gains_global.append(g_global)

        # 2) 전역보정 위에 구간별 추가보정 (전역보정 후 잔차의 구간평균)
        resid_after_global = yv - adj_global
        adj_bucket = adj_global.copy()
        for i in range(len(BUCKETS)):
            mfit = fit_m & (bid == i)
            mev = ev_m & (bid == i)
            if mfit.sum() < 30 or mev.sum() == 0:
                continue
            extra = resid_after_global[mfit].mean()
            adj_bucket[mev] = adj_bucket[mev] + extra
            g_b = sc(adj_bucket, mev) - sc(adj_global, mev)
            bucket_detail[i].append((tag, g_b, extra))
        g_bucket_all = sc(adj_bucket, ev_m) - sc(pred, ev_m)
        gains_bucket.append(g_bucket_all)

    print(f'전역보정만:        {gains_global[0]:+7.2f} / {gains_global[1]:+7.2f}   평균 {np.mean(gains_global):+.2f}')
    print(f'전역+구간보정:      {gains_bucket[0]:+7.2f} / {gains_bucket[1]:+7.2f}   평균 {np.mean(gains_bucket):+.2f}')
    print(f'구간보정 순수기여:  {np.mean(gains_bucket)-np.mean(gains_global):+.2f}')
    print()
    print('구간별 순수 기여 상세 (전역보정 이후 잔차 기준):')
    for i, (lo, hi) in enumerate(BUCKETS):
        det = bucket_detail[i]
        if not det:
            continue
        for tag, g_b, extra in det:
            print(f'  n={lo}-{hi if hi<10**8 else "+"} {tag:8s} 추가shift={extra:+.4f}  구간내이득={g_b:+.2f}')
    print()
    results[p] = (gains_global, gains_bucket)

print('=== 종합 ===')
for p in ('A', 'C'):
    gg, gb = results[p]
    print(f'fold {p}: 전역만 {np.mean(gg):+.2f} -> 전역+구간 {np.mean(gb):+.2f}  (구간기여 {np.mean(gb)-np.mean(gg):+.2f})')
