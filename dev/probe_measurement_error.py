"""제출물 쌍의 페어드 Brier 차이에 대한 '실제' 표준오차를 근사 없이 직접 계산.

d_i = (p^A_i - y_i)^2 - (p^B_i - y_i)^2 = (p^A-p^B)(p^A+p^B-2y)
SE(dbar) = sd(d_i)/sqrt(n)                      <- 독립 가정(naive)
SE_cluster: 투수 단위 클러스터 로버스트          <- 같은 투수 행끼리 상관 있으면 이게 맞음
  Var(dbar) = (1/n^2) * sum_g ( sum_{i in g} (d_i - dbar) )^2

핵심: SE는 두 예측벡터가 '얼마나 다른가'에 정비례한다. 가중치 미세조정처럼
sd(p^A-p^B)가 아주 작으면 SE도 그만큼 작아진다. 통계학자 페르소나는 모든 쌍에
동일한 generic SE(4~15점)를 적용했는데, 그건 쌍마다 다시 계산해야 하는 값이다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['season', 'pitcher_id'])
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


va = season == 2024
yv = y_all[va]
n = len(yv)
pid = raw_all.loc[va, 'pitcher_id'].to_numpy()
H = build8('A')
W = {k: float(v95[f'{k}_weight']) for k in H}
tot = sum(W.values()); W = {k: v / tot for k, v in W.items()}
blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)

# 투수 클러스터 인덱스
uniq, gidx = np.unique(pid, return_inverse=True)
n_clusters = len(uniq)


def paired_se(pA, pB):
    d = (pA - yv) ** 2 - (pB - yv) ** 2
    dbar = d.mean()
    se_naive = d.std(ddof=1) / np.sqrt(n)
    # 클러스터 로버스트
    dev = d - dbar
    gsum = np.bincount(gidx, weights=dev, minlength=n_clusters)
    var_cl = np.sum(gsum ** 2) / n ** 2
    se_cl = np.sqrt(var_cl)
    return dbar, se_naive, se_cl, np.std(pA - pB)


def report(label, pA, pB, observed_delta=None):
    dbar, se_n, se_c, sdiff = paired_se(pA, pB)
    dS = -K * dbar
    seS_n = K * se_n
    seS_c = K * se_c
    line = (f'{label:<34}{sdiff:>11.5f}{dS:>+10.2f}{seS_n:>9.2f}{seS_c:>9.2f}'
            f'{se_c/se_n:>8.2f}')
    if observed_delta is not None:
        line += f'{observed_delta:>+11.2f}{observed_delta/seS_c:>8.2f}'
    print(line)


print('실제 예측벡터로 계산한 페어드 SE (fold A 2024, n={:,}, 투수 {:,}명)'.format(n, n_clusters))
print(f'{"비교쌍":<34}{"sd(pA-pB)":>11}{"ΔScore":>10}{"SE_naive":>9}{"SE_clust":>9}{"배율":>8}'
      f'{"실측Δ":>11}{"z(실측)":>8}')
print('-' * 100)

# 1) 가중치 미세조정 (v103류: mc5 -8.6%. mc5 캐시가 없어 ordinal로 동일규모 대리)
for head, frac in [('ordinal', 0.086), ('ordinal', 0.20), ('hurdle', 0.36)]:
    W2 = dict(W)
    W2[head] = W[head] * (1 - frac)
    s = sum(W2.values())
    W2 = {k: v / s for k, v in W2.items()}
    p2 = np.clip(sum(W2[k] * H[k] for k in H), 0, 1)
    report(f'{head} 가중치 -{frac*100:.0f}%', blend, p2)

# 2) v108 = 0.97*v95 + 0.03*xgb  (실측 -1.19)
p_xgb = np.load('dev/cache_xgbrawid_A.npy')
p_v108 = np.clip(0.97 * blend + 0.03 * p_xgb, 0, 1)
report('v108 (XGB w=0.03)', blend, p_v108, observed_delta=-1.19)

# 3) 헤드 하나 제거 (v99류 donor 제거, 실측 -15.7)
W3 = {k: v for k, v in W.items() if k not in ('multires', 'condball', 'countresid')}
s = sum(W3.values()); W3 = {k: v / s for k, v in W3.items()}
p_v99 = np.clip(sum(W3[k] * H[k] for k in W3), 0, 1)
report('donor 3헤드 제거 (v99류)', blend, p_v99, observed_delta=-15.7)

# 4) 완전히 다른 모델 (persona head, 예측corr 0.96)
p_persona = np.load('dev/cache_persona_A.npy')
report('persona 단독 (corr~0.96)', blend, p_persona)

# 5) 상수 레벨 시프트 (프로브)
report('상수 shift +0.008 (프로브)', blend, np.clip(blend + 0.008, 0, 1))

print('\n[검증] 통계학자 주장: SE(ΔBS)~4e-5 -> 12~18점 / 미세조정은 3~7점')
print('       위 SE_clust 열과 비교하라.')

# 상수 시프트의 곡률이 정확히 1인지 확인 (프로브 항등식 검증)
print('\n=== 프로브 항등식 검증: 상수 shift는 곡률 V가 정확히 1인가? ===')
bs0 = np.mean((blend - yv) ** 2)
for delta in (0.004, 0.008, -0.008):
    bs = np.mean((np.clip(blend + delta, 0, 1) - yv) ** 2)
    A_true = np.mean(blend - yv)
    pred = bs0 + 2 * delta * A_true + delta ** 2
    print(f'  delta={delta:+.3f}: 실제BS={bs:.8f}  항등식예측={pred:.8f}  '
          f'차이={abs(bs-pred):.2e}  (클리핑 영향만)')
print(f'  -> V는 미지수가 아니라 정확히 1. 단일 프로브로 A가 유일하게 결정됨.')
