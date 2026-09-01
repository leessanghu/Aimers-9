"""전제 검증: 물리헤드가 '저표본 투수' 구간에서만 유효한가?
fold A(2024)에서 asof_pitcher_n 층별로 물리헤드의 기여를 따로 측정한다.
게이팅 설계는 이 결과가 참일 때만 의미가 있다."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B

meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
va = season == 2024
yv = y[va]

blend = np.load('dev/cache_v88_final_2024.npy')
phys = np.load('dev/cache_physhead_2024.npy')
resid = yv - blend

df = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['season', 'asof_pitcher_n'])
n_ = np.nan_to_num(df.loc[va, 'asof_pitcher_n'].to_numpy(np.float64), nan=0.0)

sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)
allm = np.ones(len(yv), bool)
mth = X.loc[va, 'game_month'].to_numpy()
H1 = mth <= 6
H2 = ~H1

print(f'fold A n={len(yv):,}   blend={sc(blend, allm):.2f}   물리헤드단독={sc(phys, allm):.2f}')
print(f'asof_pitcher_n: 중앙값={np.median(n_):.0f}  기존 lown_threshold=1776')

# train(<=2023)에서 산출한 분위 경계 (Rule4: test 배치 의존 금지 -> train 상수)
n_tr = np.nan_to_num(df.loc[season <= 2023, 'asof_pitcher_n'].to_numpy(np.float64), nan=0.0)
qs = np.quantile(n_tr, [0.25, 0.5, 0.75])
print(f'train 분위 경계: Q1<{qs[0]:.0f}  Q2<{qs[1]:.0f}  Q3<{qs[2]:.0f}')

d = phys - blend
print()
print('=== asof_pitcher_n 층별 물리헤드 기여 ===')
print(f'{"층":22s} {"n행":>9s} {"blend":>9s} {"corr":>9s} {"이론최대":>9s} {"H1->H2":>9s} {"H2->H1":>9s}')
strata = [
    ('Q1 저표본 (<%.0f)' % qs[0], n_ < qs[0]),
    ('Q2 (%.0f~%.0f)' % (qs[0], qs[1]), (n_ >= qs[0]) & (n_ < qs[1])),
    ('Q3 (%.0f~%.0f)' % (qs[1], qs[2]), (n_ >= qs[1]) & (n_ < qs[2])),
    ('Q4 고표본 (>=%.0f)' % qs[2], n_ >= qs[2]),
    ('--- lown(<1776)', n_ < 1776),
    ('--- high(>=1776)', n_ >= 1776),
    ('--- 전체', allm),
]
for tag, m in strata:
    if m.sum() < 100:
        continue
    dm = d[m] - d[m].mean()
    rm = resid[m]
    C = np.mean(dm * rm)
    V = np.mean(dm ** 2)
    rho = C / np.sqrt(V * rm.var()) if V > 0 else 0.0
    maxg_local = (C * C / V) * K * (m.sum() / len(yv))  # 전체 대비 기여로 환산
    gains = []
    for fit_m, ev_m in [(H1 & m, H2 & m), (H2 & m, H1 & m)]:
        if fit_m.sum() < 100 or ev_m.sum() < 100:
            gains.append(np.nan)
            continue
        Cf = np.mean((d[fit_m] - d[fit_m].mean()) * resid[fit_m])
        Vf = np.mean((d[fit_m] - d[fit_m].mean()) ** 2)
        a = Cf / Vf if Vf > 1e-12 else 0.0
        bl = blend.copy()
        bl[ev_m] = blend[ev_m] + a * (d[ev_m] - d[fit_m].mean())
        gains.append(sc(bl, ev_m) - sc(blend, ev_m))
    print(f'{tag:22s} {m.sum():>9,} {sc(blend,m):>9.1f} {rho:>+9.5f} {maxg_local:>+9.2f} '
          f'{gains[0]:>+9.2f} {gains[1]:>+9.2f}')

print()
print('=== 게이팅 블렌드 실험: 저표본 행에만 물리헤드를 w만큼 섞기 ===')
print('   preds = (1-w*gate)*blend + w*gate*phys')
for thr in (500, 1000, 1776, 3000):
    gate = (n_ < thr).astype(np.float64)
    print(f'  [thr={thr:5d}, 적용비율={gate.mean()*100:5.1f}%]')
    for w in (0.10, 0.20, 0.30):
        p_ = (1 - w * gate) * blend + w * gate * phys
        g_all = sc(p_, allm) - sc(blend, allm)
        # 정직검증: H1에서 w를 골랐다고 가정하고 H2에서 평가(여기선 w 고정이라 참고용)
        print(f'      w={w:.2f}  전체 델타={g_all:+7.2f}   '
              f'(저표본 구간만: {sc(p_, gate>0)-sc(blend, gate>0):+7.2f})')
