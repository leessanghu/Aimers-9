"""H1/H2 반쪼개기 대신, fold 전체(2024 또는 2022)로 학습한 스태킹 가중치를
'완전히 다른 연도'(교차 fold)에 그대로 적용해서 평가. 데이터 2배(월별 반쪼개기 없음),
오염 없음(다른 연도라 절대 같은 행이 없음). 이게 진짜 배포 시나리오와 동일.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50']
RIDGE_LAMBDAS = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.3]

meta = pd.read_parquet('dev/featcache_meta.parquet')
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
    mu_h = Hmat.mean(axis=0); mu_y = yy.mean()
    Hc = Hmat - mu_h; yc = yy - mu_y
    p = Hc.shape[1]
    A = Hc.T @ Hc + lam * len(yy) * np.eye(p)
    b = Hc.T @ yc
    w = np.linalg.solve(A, b)
    return w, mu_h, mu_y


def apply_stack(Hmat, w, mu_h, mu_y):
    return np.clip(mu_y + (Hmat - mu_h) @ w, 0, 1)


def sc(pp, yy):
    return 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yy) ** 2) / B)


data = {}
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    Hmat = np.column_stack([H[k] for k in HEADS])
    blend, W = fixed_blend(H)
    data[tag] = dict(y=yv, H=Hmat, blend=blend, W=W)

print(f'{"lambda":<10}{"A학습->C적용":>16}{"C학습->A적용":>16}{"평균":>10}{"기존블렌드 대비":>16}')
base_avg = (sc(data['A']['blend'], data['A']['y']) + sc(data['C']['blend'], data['C']['y'])) / 2
for lam in RIDGE_LAMBDAS:
    wA, muA, myA = ridge_fit(data['A']['H'], data['A']['y'], lam)
    wC, muC, myC = ridge_fit(data['C']['H'], data['C']['y'], lam)
    p_on_C = apply_stack(data['C']['H'], wA, muA, myA)   # A에서 배운 가중치 -> C에 적용
    p_on_A = apply_stack(data['A']['H'], wC, muC, myC)   # C에서 배운 가중치 -> A에 적용
    g_C = sc(p_on_C, data['C']['y']) - sc(data['C']['blend'], data['C']['y'])
    g_A = sc(p_on_A, data['A']['y']) - sc(data['A']['blend'], data['A']['y'])
    avgg = (g_A + g_C) / 2
    print(f'{lam:<10}{g_A:>+16.2f}{g_C:>+16.2f}{avgg:>+10.2f}{"":>16}')

print(f'\n[기존 고정블렌드 평균(A,C)] = {base_avg:.2f}')
print('\n[해석] A에서 배운 가중치를 C에, C에서 배운 가중치를 A에 그대로 이식했을 때')
print(' 둘 다 양수로 나오고 lambda에 안정적이면 -> 진짜 구조(채택 가능).')
print(' 여전히 부호가 갈리거나 lambda 민감하면 -> 헤드간 과도한 상관 때문에 못 씀(진짜 결론).')

print(f'\n=== 참고: 가중치벡터 자체 비교 ===')
w005 = {}
for tag in ('A', 'C'):
    w, _, _ = ridge_fit(data[tag]['H'], data[tag]['y'], 0.01)
    w005[tag] = w / (np.abs(w).sum() + 1e-12)
    print(f'  {tag}(lambda=0.01): ' + '  '.join(f'{h}={v:+.3f}' for h, v in zip(HEADS, w005[tag])))
print(f'  벡터간 상관 = {np.corrcoef(w005["A"], w005["C"])[0,1]:+.3f}')
