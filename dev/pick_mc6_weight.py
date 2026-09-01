"""mc6원본/구종축을 신규헤드로 얼마나 태울지 - SE 기반으로 결정.
오늘 확립한 방법: SE(ΔScore) propto sd(w*d) = w * sd(d). w를 알면 SE를 알 수 있다.
목표: 회귀예측(+11.55 또는 +5.21)이 진짜라면 z>=2로 검출 가능한 최소 w,
      동시에 틀렸을 때(XGB류처럼 -1.19/0.03 비슷한 배율로 손해) 감수할 손실 크기를 같이 보여준다.
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

uniq, gidx = np.unique(pid, return_inverse=True)


def se_at_weight(p_cand, w, label, pred_real_delta):
    d = w * (p_cand - blend)
    d_exact = d * (2 * blend + d - 2 * yv)   # (blend+d-y)^2-(blend-y)^2
    dev = d_exact - d_exact.mean()
    gsum = np.bincount(gidx, weights=dev, minlength=len(uniq))
    se_cl = K * np.sqrt(np.sum(gsum ** 2) / n ** 2)
    z_if_real = pred_real_delta / se_cl if se_cl > 0 else 0
    return se_cl, z_if_real


candidates = [
    ('mc6원본', 'dev/cache_mc6head_A.npy', 11.55, -2.28),
    ('구종축A', 'dev/cache_pitchtypehead_A.npy', 5.21, -0.85),
]

for name, path, pred_delta, foldA_local in candidates:
    p = np.load(path)
    print(f'\n=== {name} (fold A 로컬={foldA_local:+.2f}, 회귀예측 실측Δ={pred_delta:+.2f}) ===')
    print(f'{"w":>6}{"SE(추정)":>10}{"z(예측대로일때)":>16}{"손해시 예상손실(XGB배율 참고)":>24}')
    for w in (0.03, 0.05, 0.08, 0.10, 0.15, 0.20):
        se, z = se_at_weight(p, w, name, pred_delta)
        # XGB 선례: w=0.03일 때 실측 -1.19였음. 같은 비율로 손해난다면:
        xgb_scale_loss = -1.19 * (w / 0.03)
        print(f'{w:>6.2f}{se:>10.2f}{z:>16.2f}{xgb_scale_loss:>24.2f}')
