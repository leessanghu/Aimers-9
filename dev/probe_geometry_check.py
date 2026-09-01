"""기하학자 주장 검증.

(1) ||r_95||의 Var(eta) 민감도
    BS(p) = ||p-eta||^2 + E[eta(1-eta)],  E[eta(1-eta)] = rbar(1-rbar) - Var(eta)
    => ||r_95||^2 = BS(v95) - rbar(1-rbar) + Var(eta) = Var(eta) - (설명된분산)
    Var(eta)는 식별 불가능한 양이라 "12%"가 얼마나 흔들리는지 본다.
    동시에 Var(eta)의 경험적 하한들(카운트/투수/v95)을 직접 측정.

(2) '지배신호가 1차원'이라는 전제 검증
    v95 예측을 단일지표 s = w^T x 의 매끄러운 함수로 얼마나 재현할 수 있나?
    R^2가 높으면 단일지표 분해(레버A)의 전제가 성립, 낮으면 레버A는 신호를 버리는 것.
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
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                      usecols=['season', 'pitcher_id', 'balls_before', 'strikes_before'])
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
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


va = season == 2024
yv = y_all[va]
Xv = X.loc[va, FEAT].astype(np.float64)
raw = raw_all[raw_all['season'] == 2024].reset_index(drop=True)
H = build8('A')
W = {k: float(v95[f'{k}_weight']) for k in H}
t = sum(W.values()); W = {k: v / t for k, v in W.items()}
blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)

rbar = yv.mean()
var_label = rbar * (1 - rbar)
bs95 = float(np.mean((blend - yv) ** 2))
explained = var_label - bs95

print('=== (1) 기본량 (fold A 2024) ===')
print(f'  rbar={rbar:.5f}   rbar(1-rbar)={var_label:.6f}   BSref={B:.6f}')
print(f'  BS(v95)={bs95:.6f}   v95가 설명한 분산 = {explained:.6f}')

print('\n=== Var(eta)의 경험적 하한들 (어떤 모델이든 설명한 분산은 Var(eta)의 하한) ===')
# 카운트만으로 설명되는 분산
cs = (raw['balls_before'] * 4 + raw['strikes_before']).to_numpy()
dfc = pd.DataFrame({'cs': cs, 'y': yv})
gm = dfc.groupby('cs')['y'].agg(['mean', 'count'])
var_count = float(np.average((gm['mean'] - rbar) ** 2, weights=gm['count']))
# 투수만으로 (축소 없이 = 상향편향, 축소 적용 = 보수적)
gp = pd.DataFrame({'p': raw['pitcher_id'].to_numpy(), 'y': yv}).groupby('p')['y'].agg(['mean', 'count'])
var_pitch_raw = float(np.average((gp['mean'] - rbar) ** 2, weights=gp['count']))
# 표본노이즈 제거(ANOVA 방식): Var_true = Var_obs - E[within/n]
noise = float(np.average(var_label / gp['count'], weights=gp['count']))
var_pitch_adj = max(var_pitch_raw - noise, 0.0)
print(f'  카운트(cs)만                = {var_count:.6f}  ({var_count/var_label*100:5.2f}% of label var)')
print(f'  투수 raw(상향편향)          = {var_pitch_raw:.6f}')
print(f'  투수 노이즈보정(ICC 추정)   = {var_pitch_adj:.6f}  (ICC={var_pitch_adj/var_label*100:.2f}%)')
print(f'  v95 (실제 달성)             = {explained:.6f}  <- 이게 Var(eta)의 가장 강한 하한')

print('\n=== "놓친 신호의 88%가 좌표계 밖" 주장의 Var(eta) 민감도 ===')
print(f'{"가정 Var(eta)":>14}{"||r95||^2":>12}{"||r95||":>10}{"도달가능비율":>14}{"남은잠재점수":>13}')
for ve in (0.00280, 0.00300, 0.0032, 0.0035, 0.0040, 0.0050, 0.0070):
    r2 = ve - explained
    if r2 < 0:
        print(f'{ve:>14.5f}{"불가능(<설명분산)":>12}')
        continue
    rn = np.sqrt(r2)
    reach = 0.0032 / rn if rn > 0 else np.inf   # 기하학자의 proj 길이 가정
    print(f'{ve:>14.5f}{r2:>12.6f}{rn:>10.4f}{min(reach,1.0)*100:>13.1f}%{K*r2:>13.0f}')

print('\n  [주의] 기하학자는 Var(eta)=0.0035를 가정해 ||r95||=0.027, 도달가능 12%를 얻었다.')
print('  하지만 Var(eta)는 식별 불가능한 양이고, 위 표처럼 가정에 따라 도달가능비율이')
print('  20%~100%까지 요동친다. "88%가 좌표계 밖"은 측정된 사실이 아니라 가정의 재진술이다.')

print('\n=== (2) "지배신호는 1차원"인가? v95 예측의 단일지표 재현율 ===')
lg = np.log(np.clip(blend, 1e-6, 1 - 1e-6) / (1 - np.clip(blend, 1e-6, 1 - 1e-6)))
Xs = Xv.to_numpy(np.float64)
Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)
mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
Z = (Xs - mu) / sd
ridge = Ridge(alpha=10.0).fit(Z, lg)
s = Z @ ridge.coef_
# s의 매끄러운 함수로 v95 로짓을 얼마나 재현하나 (200분위 구간평균 = 비모수 1D 최적)
qs = np.quantile(s, np.linspace(0, 1, 201)[1:-1])
bins = np.digitize(s, qs)
fit1d = pd.Series(lg).groupby(bins).transform('mean').to_numpy()
ss_res = float(np.mean((lg - fit1d) ** 2))
ss_tot = float(np.var(lg))
print(f'  단일지표 s로 v95 로짓 재현 R^2 = {1 - ss_res/ss_tot:.4f}')
print(f'  (R^2가 0.95+면 "1차원 지배" 전제 성립. 낮으면 레버A는 신호를 버리는 것)')

# 확률 스케일에서도
p1d = 1 / (1 + np.exp(-fit1d))
bs_1d = float(np.mean((p1d - yv) ** 2))
print(f'  1D 단일지표 예측만의 BSS = {1e5*(1-bs_1d/B):.1f}   (v95 = {1e5*(1-bs95/B):.1f})')
print(f'  -> 1D로 떨어뜨렸을 때 잃는 점수 = {1e5*(1-bs95/B) - 1e5*(1-bs_1d/B):.1f}점')
