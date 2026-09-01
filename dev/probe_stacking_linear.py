"""선형 스태킹 메타러너: 수동 고정가중치 대신 8개 헤드의 최적 결합계수를
'학습'으로 직접 찾는다. Ridge 정규화(헤드간 상관 0.9+라 필수) + 중심화 + 무절편
(H1/H2 절편오염 버그 재발 방지, [[h1h2-intercept-contamination]]).

절차(양쪽 fold 모두):
  1) H1에서 8헤드 ridge회귀로 가중치 학습 -> H2에서 평가, 반대방향도 동일
  2) 같은 H2/H1 구간에서 '기존 고정가중치 블렌드' 점수와 직접 비교(사과대사과)
  3) fold A와 fold C에서 학습된 가중치벡터끼리 상관 -> 재현되는 구조인지 노이즈인지 판별
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50']
RIDGE_LAMBDAS = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3]

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
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


def fixed_blend(H):
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t = sum(W.values()); W = {k: v / t for k, v in W.items()}
    return np.clip(sum(W[k] * H[k] for k in H), 0, 1), W


def ridge_fit(Hmat, yy, lam):
    """중심화 + 무절편 ridge. Hmat: (n,8). 반환: 가중치(8,), 중심값들"""
    mu_h = Hmat.mean(axis=0)
    mu_y = yy.mean()
    Hc = Hmat - mu_h
    yc = yy - mu_y
    p = Hc.shape[1]
    A = Hc.T @ Hc + lam * len(yy) * np.eye(p)
    b = Hc.T @ yc
    w = np.linalg.solve(A, b)
    return w, mu_h, mu_y


def apply_stack(Hmat, w, mu_h, mu_y):
    return np.clip(mu_y + (Hmat - mu_h) @ w, 0, 1)


def sc(pp, yy):
    return K * (1 - np.mean((np.clip(pp, 0, 1) - yy) ** 2) / (yy.mean() * (1 - yy.mean())) * (yy.mean() * (1 - yy.mean())) / B) if False else 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yy) ** 2) / B)


results = {}
weight_vecs = {}
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    Hmat = np.column_stack([H[k] for k in HEADS])
    blend, W = fixed_blend(H)

    Xv = X.loc[va]
    mth = Xv['game_month'].to_numpy()
    H1 = mth <= 6; H2 = ~H1

    print(f'\n=== fold {tag} ===')
    print(f'  기존 고정블렌드: 전체BSS={sc(blend, yv):.2f}  H1구간={sc(blend[H1], yv[H1]):.2f}  H2구간={sc(blend[H2], yv[H2]):.2f}')

    best_lam, best_avg = None, -1e18
    for lam in RIDGE_LAMBDAS:
        gains = []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            w, mu_h, mu_y = ridge_fit(Hmat[fit_m], yv[fit_m], lam)
            p_stack = apply_stack(Hmat[ev_m], w, mu_h, mu_y)
            gains.append(sc(p_stack, yv[ev_m]) - sc(blend[ev_m], yv[ev_m]))
        avgg = np.mean(gains)
        print(f'    ridge_lambda={lam:<8} H1->H2={gains[0]:+7.2f}  H2->H1={gains[1]:+7.2f}  평균={avgg:+7.2f}')
        if avgg > best_avg:
            best_avg = avgg; best_lam = lam

    # 최적 람다로 전체 fold 데이터에 학습한 가중치벡터 저장 (재현성 비교용)
    w_full, mu_h_full, mu_y_full = ridge_fit(Hmat, yv, best_lam)
    weight_vecs[tag] = w_full / (np.abs(w_full).sum() + 1e-12)
    results[tag] = best_avg
    print(f'  [best] lambda={best_lam}  평균이득={best_avg:+.2f}')
    print(f'  학습된 상대가중치(절대값정규화): ' +
          '  '.join(f'{h}={v:+.3f}' for h, v in zip(HEADS, weight_vecs[tag])))
    print(f'  기존 수동가중치(정규화): ' + '  '.join(f'{h}={W[h]:+.3f}' for h in HEADS))

print(f'\n=== 재현성: fold A vs fold C 학습가중치벡터 상관 ===')
corr = np.corrcoef(weight_vecs['A'], weight_vecs['C'])[0, 1]
print(f'  corr = {corr:+.3f}  (높으면 진짜 구조, 낮으면 fold마다 다른 노이즈)')
