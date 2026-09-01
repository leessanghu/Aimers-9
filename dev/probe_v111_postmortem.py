"""v111 실측(-1.169) 사후분석.
바꾼 게 두 가지 동시(배깅 K=1->5, level_shift +0.00097) -> 분리진단.

핵심 질문: -1.169가 '진짜 실패'인가, 기대값 +1.56 근처의 '노이즈 범위 안'인가?

배깅쪽 SE 추정: 시드쌍 캐시로 잰 sigma_k(단일시드 표준편차)를 이용해
'K=1(v95, 특정 시드 하나) vs K=5(평균)' 차이벡터의 sd를 유도한다.
  diff = bagged_avg - f_specific,  f_i iid(sigma^2)
  Var(diff) = Var((1/5)sum f_i - f_1) = (4/5)^2 sigma^2 + (1/5)^2*4*sigma^2 = 0.8*sigma^2
  -> sd(diff) = sqrt(0.8)*sigma  (기존 sigma와 거의 같은 크기! 별로 안 줄어든다)
이 sd(diff)를 헤드별 가중치로 합성해서, 오늘 확립한 페어드-SE 공식으로 실제 SE(ΔScore) 추정.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B

DELTA_OBSERVED = 1102.487862473 - 1103.6568315036
LEVEL_DELTA = 0.00097  # level_shift 변경폭 (-0.00127 -> -0.00030)

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['season', 'pitcher_id'])
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

SEED_PAIRS = {
    'multires':   'dev/idea13_cache/{t}_multires_s{s}.npy',
    'ordinal':    'dev/idea13_cache/{t}_ordinal_s{s}.npy',
    'midother':   'dev/idea46_cache/{t}_midother_s{s}.npy',
    'condball':   'dev/idea54_cache/{t}_cond_ball_s{s}.npy',
    'countresid': 'dev/idea54_cache/{t}_count_resid_s{s}.npy',
    'future50':   'dev/idea54_cache/{t}_future50_multi_s{s}.npy',
}
ALL_HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
             'condball', 'countresid', 'future50']
W = {k: float(v95[f'{k}_weight']) for k in ALL_HEADS}
tot = sum(W.values()); W = {k: v / tot for k, v in W.items()}


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


print(f'=== 관측된 실측 변화 ===')
print(f'  v95=1103.6568  v111=1102.4879  Δ={DELTA_OBSERVED:+.3f}')
print(f'  기대값이었던 +1.56 (배깅+1.18 + 레벨+0.38) 과의 괴리 = {DELTA_OBSERVED-1.56:+.3f}\n')

for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    n = len(yv)
    pid = raw_all.loc[va, 'pitcher_id'].to_numpy()
    H = build8(tag)
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)

    # 배깅 diff벡터 합성: sqrt(0.8)*sigma_k, 헤드별 가중치 곱해서 벡터로 근사 결합
    # (실제 상관구조를 모르니 부호를 시드쌍 실측(p1-p2)으로 근사 대리)
    diff_total = np.zeros(n)
    for head, pat in SEED_PAIRS.items():
        p1 = np.load(pat.format(t=tag, s=42))
        p2 = np.load(pat.format(t=tag, s=7))
        raw_diff = p1 - p2                      # sigma*sqrt(2) 크기의 실제 관측된 차이벡터
        scale = np.sqrt(0.8 / 2.0)               # sqrt(0.8)*sigma 로 스케일 맞추기 (raw_diff의 sd = sigma*sqrt2)
        diff_total += W[head] * raw_diff * scale

    resid = yv - blend
    C = float(np.mean(resid * diff_total)); V = float(np.mean(diff_total ** 2))
    sd_diff = np.std(diff_total)

    # 클러스터(투수단위) 로버스트 SE, 오늘 확립한 페어드-SE 방법론 그대로
    d_i = diff_total * (2 * (blend - yv) + diff_total)   # (pA-y)^2-(pB-y)^2 근사식의 부호 반대 조심
    # 정확식: d = (pB-y)^2-(pA-y)^2 = (pB-pA)(pB+pA-2y). diff_total = "pB-pA" 방향(=bag-orig)로 정의.
    d_exact = diff_total * (2 * blend + diff_total - 2 * yv)
    uniq, gidx = np.unique(pid, return_inverse=True)
    dev = d_exact - d_exact.mean()
    gsum = np.bincount(gidx, weights=dev, minlength=len(uniq))
    se_cl = np.sqrt(np.sum(gsum ** 2) / n ** 2)
    seS_cl = K * se_cl
    dS = K * d_exact.mean()

    print(f'--- fold {tag} ---')
    print(f'  배깅 차이벡터 sd(bag-orig) 근사 = {sd_diff:.5f}')
    print(f'  로컬 추정 배깅효과(방향성 참고용) = {dS:+.2f}점')
    print(f'  이 크기 변화의 실제 SE(클러스터) = {seS_cl:.2f}점')
    print(f'  -> 배깅 변화만으로도 SE가 {seS_cl:.2f}점이면, 기대값(+1.18)과 실측 -1.169는')
    print(f'     z = {(DELTA_OBSERVED-1.18)/seS_cl:+.2f} (레벨보정 +0.38 별도, 대략치)\n')
