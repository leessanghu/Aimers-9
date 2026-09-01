"""전역 레벨보정 축소(-0.0013)가 타당했는지 재검증.
(1) fold A에서 월별 잔차가 안정적인가(=상수 shift가 안전한가)
(2) fold A/C에서 '연도간 드리프트 -> 잔여편차' 흡수율이 일관되는가"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
UNC = 0.249807

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'])
tot = sum(W.values())
W = {k: v / tot for k, v in W.items()}   # 8헤드 정규화(mc5/ingame 캐시가 A만 있어 제외)
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


rates = {int(s): y[season == s].mean() for s in sorted(pd.unique(season))}

print('=== (1) fold별 연도간 드리프트 vs 실제 잔여편차 ===')
print(f'{"fold":>5s} {"train마지막":>10s} {"검증연도":>8s} {"드리프트":>10s} {"실제편차D":>10s} {"흡수율":>8s} {"보정이득":>9s}')
for p, last_tr, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    va = season == vs
    yv = y[va]
    H = load8(p)
    pred = sum(W[k] * H[k] for k in W)
    D = pred.mean() - yv.mean()
    drift = rates[vs] - rates[last_tr]
    ratio = D / (-drift) if abs(drift) > 1e-9 else float('nan')
    gain = (1e5 / UNC) * D * D
    print(f'{p:>5s} {last_tr:>10d} {vs:>8d} {drift:>+10.5f} {D:>+10.5f} {ratio:>8.1%} {gain:>+9.2f}')

print()
print('=== (2) fold A 월별 잔차(예측-실제): 상수 shift가 안전한가 ===')
va = season == 2024
yv = y[va]
H = load8('A')
pred = sum(W[k] * H[k] for k in W)
mth = X.loc[va, 'game_month'].to_numpy()
Dall = pred.mean() - yv.mean()
print(f'  전체 D = {Dall:+.5f}')
for m in sorted(np.unique(mth)):
    mm = mth == m
    if mm.sum() < 500:
        continue
    d = pred[mm].mean() - yv[mm].mean()
    print(f'    {int(m):2d}월 n={mm.sum():>7,}  D={d:+.5f}  (전체D 대비 {d-Dall:+.5f})')

print()
print('=== (3) fold A: 전체D로 상수보정시 H1/H2 각각 이득 ===')
sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / UNC)
H1 = mth <= 6
H2 = ~H1
for msk, tag in [(H1, 'H1(1-6월)'), (H2, 'H2(7-12월)'), (np.ones(len(yv), bool), '전체')]:
    g = sc(pred - Dall, msk) - sc(pred, msk)
    print(f'  {tag:12s} 이득 = {g:+.2f}')

print()
print('=== (4) H1에서 잰 D를 H2에 적용(정직 전이) ===')
d_h1 = pred[H1].mean() - yv[H1].mean()
d_h2 = pred[H2].mean() - yv[H2].mean()
print(f'  H1에서 잰 D={d_h1:+.5f} -> H2 적용 이득 = {sc(pred-d_h1, H2)-sc(pred, H2):+.2f}')
print(f'  H2에서 잰 D={d_h2:+.5f} -> H1 적용 이득 = {sc(pred-d_h2, H1)-sc(pred, H1):+.2f}')
