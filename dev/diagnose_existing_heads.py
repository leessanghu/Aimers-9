"""기존 v95 10개 헤드 진단.
1) 각 헤드의 실제 신호 대비 가중치가 적절한가 (leave-one-out 재확인, fold A/C)
2) 각 헤드가 mc6와 얼마나 겹치는가 (d상관) - 오늘 발견한 '중복이면 손해' 법칙 적용
3) 시드분산 있는 헤드들의 남은 배깅 여지 (mc6처럼 저평가됐을 수 있음)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50']


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


for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in HEADS}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in HEADS), 0, 1)
    sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / B_)
    d_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy') - blend; d_mc6 -= d_mc6.mean()

    print(f'\n{"="*90}\n=== fold {tag} ({vs}) ===\n{"="*90}')
    print(f'{"헤드":<13}{"w":>7}{"단독BSS":>9}{"제거시":>9}{"기여":>8}{"블렌드상관":>10}{"mc6와상관":>10}{"시드std":>9}')
    for k in HEADS:
        W2 = {j: W[j] for j in HEADS if j != k}
        s2 = sum(W2.values()); W2 = {j: v / s2 for j, v in W2.items()}
        p_wo = np.clip(sum(W2[j] * H[j] for j in W2), 0, 1)
        contrib = sc(blend) - sc(p_wo)
        corr_blend = np.corrcoef(H[k], blend)[0, 1]
        d_k = H[k] - blend; d_k = d_k - d_k.mean()
        corr_mc6 = float(np.mean(d_k * d_mc6) / np.sqrt(np.mean(d_k**2) * np.mean(d_mc6**2) + 1e-18))

        # 시드분산 (있는 헤드만)
        seed_std = np.nan
        try:
            if k == 'base':
                p1 = avg([f'dev/phase90_cache/{tag}_base_d6.npy'])
                p2 = avg([f'dev/phase90_cache/{tag}_base_d8.npy'])
            elif k in ('multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50'):
                paths = {
                    'multires': f'dev/idea13_cache/{tag}_multires_s',
                    'ordinal': f'dev/idea13_cache/{tag}_ordinal_s',
                    'midother': f'dev/idea46_cache/{tag}_midother_s',
                    'condball': f'dev/idea54_cache/{tag}_cond_ball_s',
                    'countresid': f'dev/idea54_cache/{tag}_count_resid_s',
                    'future50': f'dev/idea54_cache/{tag}_future50_multi_s',
                }[k]
                p1 = np.load(paths + '42.npy'); p2 = np.load(paths + '7.npy')
            else:
                p1 = p2 = None
            if p1 is not None:
                seed_std = float(np.std(p1 - p2))
        except FileNotFoundError:
            pass

        print(f'{k:<13}{W[k]:>7.4f}{sc(H[k]):>9.1f}{sc(p_wo):>9.1f}{contrib:>+8.2f}'
              f'{corr_blend:>10.4f}{corr_mc6:>+10.4f}'
              f'{seed_std if not np.isnan(seed_std) else 0:>9.5f}')

print(f'\n{"="*90}')
print('[해석 가이드]')
print(' - "기여"가 크게 음수인데 오늘 실측(v99/v101)으로 이미 "빼면 손해" 확정된 헤드는')
print('   로컬-실측 역전 사례이므로 무시(donor-heads-mattered-real 참고)')
print(' - "mc6와상관"이 높은(>0.5) 헤드는 mc6와 겹쳐서 미래 가중치 조정시 우선 후보')
print(' - "시드std"가 mc6의 sigma=0.00757 대비 크면서 가중치도 있는 헤드는')
print('   추가 시드배깅 여지가 있을 수 있음(단, w가 작으면 효과도 작음: 이득=K*w^2*sigma^2)')
