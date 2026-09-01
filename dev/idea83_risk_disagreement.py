"""v88 risk(mc5 기반)와 다른 모델(ordinal)이 만드는 '위험 신호'가 얼마나 다른 행을
가리키는지 확인. mc5-risk와 ordinal-risk의 상관이 낮다면(기존 헤드간 상관 0.9+와
달리), 두 risk가 불일치하는 행 = 모델들이 서로 다르게 확신하는 불확실 지대 =
아직 안 써먹은 위험신호일 수 있다.

ordinal = P(not reverse) * P(not middle|not reverse) * P(success|나머지) 캐스케이드.
ordinal_risk = 1 - P(not reverse)*P(not middle|not reverse) = P(reverse or middle) 근사.
mc5_risk = P(middle)+P(reverse) (기존 v88 risk, 11-class 기준 class 9,10).

fold A에서: (1) 두 risk의 상관 (2) 불일치 행에서 잔차가 큰지 (3) 새 조합(둘 다 높을 때만
위험판정)이 mc5단독보다 잔차를 더 잘 가르는지 정직하게(H1<->H2) 검증.
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')

# v88 원본 블렌드 재구성 (10헤드 그대로, 변경 없음)
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
p = 'A'
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
P11 = np.load('dev/idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
v88_raw = sum(W[k] * H[k] for k in W)

# 기존 risk (mc5 기반, class 9=middle,10=reverse)
mc5_risk = P11[:, [9, 10]].sum(axis=1)

# ordinal_risk: ordinal 헤드 자체는 최종 성공확률(캐스케이드 곱)만 캐시돼있어서,
# stage별 확률을 따로 못 얻음. 대안: ordinal 예측이 낮을수록(=위험할수록) 큰 신호로 근사.
# ordinal은 P(not-rev)*P(not-mid|not-rev)*P(succ|나머지) 이므로 (1-ordinal예측)이 곧 위험도.
ordinal_risk = 1.0 - H['ordinal']

sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
resid = yv - v88_raw

print(f'corr(mc5_risk, ordinal_risk) = {np.corrcoef(mc5_risk, ordinal_risk)[0,1]:.4f}')
print(f'corr(mc5_risk, y)            = {np.corrcoef(mc5_risk, yv)[0,1]:+.4f}')
print(f'corr(ordinal_risk, y)        = {np.corrcoef(ordinal_risk, yv)[0,1]:+.4f}')
print()

# 불일치도: 두 risk를 표준화해서 차이의 크기
mz = (mc5_risk - mc5_risk.mean()) / mc5_risk.std()
oz = (ordinal_risk - ordinal_risk.mean()) / ordinal_risk.std()
disagree = np.abs(mz - oz)
print(f'corr(disagree, |resid|) = {np.corrcoef(disagree, np.abs(resid))[0,1]:+.4f}  (양수면 불일치=오차큰 곳)')
print()

H1 = mth <= 6; H2 = ~H1


def eval_axis(name, axis, nbin=20):
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        edges = np.unique(np.quantile(axis[fit_m], np.linspace(0, 1, nbin + 1)))
        if len(edges) < 3:
            gains.append(0.0); continue
        edges = edges.astype(float); edges[0] -= 1e-9; edges[-1] += 1e-9
        bf_ = np.clip(np.digitize(axis[fit_m], edges) - 1, 0, len(edges) - 2)
        be = np.clip(np.digitize(axis[ev_m], edges) - 1, 0, len(edges) - 2)
        rf = resid[fit_m]; gl = rf.mean()
        cmap = np.zeros(len(edges) - 1)
        for b in range(len(edges) - 1):
            m = bf_ == b
            if m.sum() >= 500:
                cmap[b] = rf[m].mean() - gl
        adj = v88_raw.copy(); adj[ev_m] = v88_raw[ev_m] + cmap[be]
        gains.append(sc(adj, ev_m) - sc(v88_raw, ev_m))
    return gains


print('=== 평균중립 구간보정 순수기여 (전역레벨 제외) ===')
for name, axis in [
    ('mc5_risk(기존)', mc5_risk),
    ('ordinal_risk(신규)', ordinal_risk),
    ('disagree(불일치도)', disagree),
    ('mc5+ordinal 곱', mc5_risk * ordinal_risk),
    ('mc5+ordinal 합', mc5_risk + ordinal_risk),
    ('둘다 높을때만(min)', np.minimum(mz, oz)),
]:
    g = eval_axis(name, axis)
    print(f'  {name:20s} H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')
