"""mc6 계층분해(3서브헤드 평균) 프로덕션 가중치 결정.
rho가 fold A/C 둘 다 음수 -> 최적가중치는 음수(=빼는 방향). SE 기반으로 크기 결정.
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

p_avg = np.mean([np.load(f'dev/cache_mc6h_{h}_A.npy')
                 for h in ('headA_wild', 'headB_ball', 'headC_strike')], axis=0)
d0 = p_avg - blend
dc = d0 - d0.mean()
resid = yv - blend
C = float(np.mean(dc * resid)); V = float(np.mean(dc ** 2))
s_star = C / V   # BS_new = BS_old - 2sC + s^2V (resid=y-blend 컨벤션) -> s* = C/V
print(f'C={C:+.3e}  V={V:.6f}  s*(최적, resid=y-blend 컨벤션) = {s_star:+.4f}')
print(f'(C<0 -> s*<0, 즉 빼는 방향이 최적)\n')

uniq, gidx = np.unique(pid, return_inverse=True)


def se_at_weight(w):
    d = w * d0
    d_exact = -d * (2 * (blend - yv) + d)   # BS변화 = 2*w*(blend-y)*d0 + ... 부호주의, 직접계산
    # 직접: BS_new-BS_old = ((blend+d)-y)^2-(blend-y)^2 = 2d(blend-y)+d^2
    d_exact = 2 * d * (blend - yv) + d ** 2
    dev = d_exact - d_exact.mean()
    gsum = np.bincount(gidx, weights=dev, minlength=len(uniq))
    se_cl = K * np.sqrt(np.sum(gsum ** 2) / n ** 2)
    return se_cl


print(f'{"w":>8}{"SE":>10}{"기대이득(2sC-s^2V 항등식)":>26}')
for w in (-0.01, -0.02, -0.03, -0.05, -0.08, -0.10, -0.15):
    se = se_at_weight(w)
    ev = -K * (-2 * w * C + w ** 2 * V)   # BS_new-BS_old = -2wC+w^2V (resid conv) -> Score변화=-K*그값
    print(f'{w:>8.2f}{se:>10.2f}{ev:>26.2f}')
