"""XGB/LGBM을 '전역 스칼라 가중치'가 아니라 '구간별 가중치'로 넣으면 얼마나 더 먹나?

Cj = (1/N)Σ_{i∈Rj}(p-y)d ,  Vj = (1/N)Σ_{i∈Rj} d²   (ΣCj=C, ΣVj=V)
  전역 최대이득   = K·C²/V
  구간별 최대이득 = K·Σ Cj²/Vj   (≥ 전역, 코시-슈바르츠)
  H = 구간별/전역  <- 로컬 C가 λ배 부풀려져도 λ²가 약분돼 H는 불변!
따라서 실측 구간별 상한 ≈ H × 0.34점 (0.34 = v108 프로브로 역산한 전역 상한)

[필수] 유한표본에서는 아무 분할이나 H>1이 나온다 -> 같은 크기 랜덤분할 대조군 필수.
[필수] fold A/C에서 sj* 부호가 일치해야 진짜 구조. 아니면 노이즈.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
GLOBAL_REAL_MAXGAIN = 0.34   # v108 프로브 역산값(foldA V 기준, 가장 관대한 쪽)

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
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


def load_fold(tag, vs):
    va = season == vs
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t = sum(W.values()); W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    Xv = X.loc[va].reset_index(drop=True)
    raw = raw_all[raw_all['season'] == vs].reset_index(drop=True)
    tr = season <= (vs - 1)
    cnt = pd.Series(raw_all.loc[tr, 'pitcher_id'].to_numpy()).value_counts()
    napp = raw['pitcher_id'].map(cnt).fillna(0).to_numpy(np.float64)
    return dict(y=y[va], blend=blend, X=Xv, raw=raw, napp=napp,
                xgb=np.load(f'dev/cache_xgbrawid_{tag}.npy'),
                lgbm=np.load(f'dev/cache_lgbmrawid_{tag}.npy'))


def analyze(p, blend, yv, labels):
    """labels: 정수배열(구간 id). 반환 (H, 전역이득, 구간별이득, 구간별 sj*)"""
    d = p - blend
    N = len(yv)
    e = blend - yv
    C = float(np.sum(e * d) / N); V = float(np.sum(d * d) / N)
    g_glob = K * C ** 2 / V
    tot = 0.0; sj = {}
    for j in np.unique(labels):
        m = labels == j
        Cj = float(np.sum(e[m] * d[m]) / N); Vj = float(np.sum(d[m] * d[m]) / N)
        if Vj <= 1e-15:
            continue
        tot += K * Cj ** 2 / Vj
        sj[int(j)] = (-Cj / Vj, int(m.sum()))
    return tot / g_glob, g_glob, tot, sj


FOLDS = {t: load_fold(t, v) for t, v in [('A', 2024), ('C', 2022)]}

def make_parts(F):
    Xv, raw, napp = F['X'], F['raw'], F['napp']
    n0 = (Xv['asof_pitcher_n'].to_numpy() == 0)
    c2 = (Xv['count_state'].to_numpy() == 2)
    gf = (raw['game_type'] == 'F').to_numpy()
    q = np.digitize(napp, np.quantile(napp[napp > 0], [.25, .5, .75]))
    q = np.where(napp == 0, 4, q)
    parts = {
        '신인(n=0) vs 나머지': n0.astype(int),
        '0-2카운트 vs 나머지': c2.astype(int),
        '퓨처스(F) vs 나머지': gf.astype(int),
        '투수등장빈도 5분할': q,
        '약점3종 우선순위 4분할': np.where(n0, 1, np.where(c2, 2, np.where(gf, 3, 0))),
        '예측확률 십분위': np.digitize(F['blend'], np.quantile(F['blend'], np.arange(.1, 1, .1))),
    }
    return parts


for model in ('xgb', 'lgbm'):
    print(f'\n{"="*78}\n### {model.upper()} raw-ID : 구간별 가중치의 추가 헤드룸 H\n{"="*78}')
    print(f'{"분할":<24}{"H(foldA)":>10}{"H(foldC)":>10}{"H(대조A)":>10}{"H(대조C)":>10}'
          f'{"실측상한추정":>13}')
    for name in make_parts(FOLDS['A']).keys():
        row = {}
        for tag in ('A', 'C'):
            F = FOLDS[tag]
            lab = make_parts(F)[name]
            h, gg, gt, sj = analyze(F[model], F['blend'], F['y'], lab)
            # 대조군: 같은 구간크기 분포를 갖는 랜덤분할 (5회 평균)
            rng = np.random.RandomState(11)
            hs = []
            for _ in range(5):
                hs.append(analyze(F[model], F['blend'], F['y'], rng.permutation(lab))[0])
            row[tag] = (h, float(np.mean(hs)), sj)
        hA, cA, sjA = row['A']; hC, cC, sjC = row['C']
        # 대조군 초과분만 실제 헤드룸으로 인정
        excess = max(0.0, min(hA - cA, hC - cC))
        est = (1.0 + excess) * GLOBAL_REAL_MAXGAIN
        print(f'{name:<24}{hA:>10.2f}{hC:>10.2f}{cA:>10.2f}{cC:>10.2f}{est:>13.2f}점')

    # 부호 재현성: 가장 유망해 보이는 분할에 대해 fold A/C sj* 부호 비교
    print(f'\n  [부호 재현성] 약점3종 4분할의 구간별 최적가중치 sj*')
    names = {0: '나머지', 1: '신인n=0', 2: '0-2카운트', 3: '퓨처스F'}
    sA = analyze(FOLDS['A'][model], FOLDS['A']['blend'], FOLDS['A']['y'],
                 make_parts(FOLDS['A'])['약점3종 우선순위 4분할'])[3]
    sC = analyze(FOLDS['C'][model], FOLDS['C']['blend'], FOLDS['C']['y'],
                 make_parts(FOLDS['C'])['약점3종 우선순위 4분할'])[3]
    for j in sorted(names):
        a = sA.get(j); c = sC.get(j)
        if a is None or c is None:
            continue
        ok = 'O' if np.sign(a[0]) == np.sign(c[0]) else 'X'
        print(f'    {names[j]:<10} foldA s*={a[0]:+7.3f}(n={a[1]:>6,})   '
              f'foldC s*={c[0]:+7.3f}(n={c[1]:>6,})   부호일치={ok}')
