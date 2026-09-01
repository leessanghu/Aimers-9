"""세 가설 통합 진단.

Krogh-Vedelsby ambiguity decomposition (제곱손실에서 '정확한 항등식'):
    BS(f_bar) = sum_i w_i * BS(f_i)  -  sum_i w_i * E[(f_i - f_bar)^2]
                 ^개별평균오차            ^ambiguity(다양성이 벌어주는 이득)

이게 세 가설을 동시에 정량화한다:
  H1(헤드 너무 많다): leave-one-head-out으로 각 헤드의 실제 기여 측정
  H2(피처 비슷비슷): 헤드간 상관/실효랭크 -> 다양성 부족의 직접 측정치
  H3(앙상블 부족):   ambiguity가 개별오차 대비 얼마나 작은지 + 더 벌 수 있는 상한

마지막에 '100등(+27점)까지 가려면 ambiguity가 얼마나 커져야 하는가'를 역산한다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B

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
    t = sum(W.values()); W = {k: v / t for k, v in W.items()}
    fbar = np.clip(sum(W[k] * H[k] for k in HEADS), 0, 1)

    sc = lambda p: 1e5 * (1 - np.mean((np.clip(p, 0, 1) - yv) ** 2) / B)
    bs = lambda p: float(np.mean((np.clip(p, 0, 1) - yv) ** 2))

    print(f'\n{"="*82}')
    print(f'=== fold {tag} ({vs}) — Ambiguity Decomposition ===')
    print(f'{"="*82}')

    indiv_bs = {k: bs(H[k]) for k in HEADS}
    mean_indiv = sum(W[k] * indiv_bs[k] for k in HEADS)
    ambiguity = sum(W[k] * float(np.mean((np.clip(H[k], 0, 1) - fbar) ** 2)) for k in HEADS)
    bs_ens = bs(fbar)

    print(f'  개별오차 가중평균 E[BS_i]  = {mean_indiv:.6f}  (점수환산 {1e5*(1-mean_indiv/B):8.1f})')
    print(f'  ambiguity(다양성 이득)     = {ambiguity:.6f}  (점수환산 {K*ambiguity:+8.1f})')
    print(f'  앙상블 BS                  = {bs_ens:.6f}  (점수환산 {sc(fbar):8.1f})')
    print(f'  항등식 검증: {mean_indiv:.8f} - {ambiguity:.8f} = {mean_indiv-ambiguity:.8f}'
          f'  vs 실제 {bs_ens:.8f}  차이={abs(mean_indiv-ambiguity-bs_ens):.2e}')

    print(f'\n  --- H1: 각 헤드의 개별 성능과 실제 기여 (leave-one-out) ---')
    print(f'{"헤드":<13}{"w":>8}{"단독BSS":>10}{"제거시BSS":>11}{"기여":>9}{"vs블렌드상관":>13}')
    for k in HEADS:
        W2 = {j: W[j] for j in HEADS if j != k}
        s2 = sum(W2.values()); W2 = {j: v / s2 for j, v in W2.items()}
        p_wo = np.clip(sum(W2[j] * H[j] for j in W2), 0, 1)
        contrib = sc(fbar) - sc(p_wo)
        corr = np.corrcoef(H[k], fbar)[0, 1]
        print(f'{k:<13}{W[k]:>8.4f}{sc(H[k]):>10.1f}{sc(p_wo):>11.1f}{contrib:>+9.2f}{corr:>13.4f}')

    print(f'\n  --- H2/H3: 헤드간 상관구조 (다양성의 직접 측정) ---')
    M = np.column_stack([H[k] for k in HEADS])
    Cm = np.corrcoef(M.T)
    off = Cm[np.triu_indices(len(HEADS), 1)]
    print(f'  헤드간 평균상관 = {off.mean():.4f}   최소 = {off.min():.4f}   최대 = {off.max():.4f}')
    ev = np.linalg.eigvalsh(Cm)[::-1]
    eff_rank = (ev.sum() ** 2) / (ev ** 2).sum()          # participation ratio
    print(f'  상관행렬 고유값 = {np.array2string(ev, precision=3, floatmode="fixed")}')
    print(f'  실효랭크(participation ratio) = {eff_rank:.2f} / {len(HEADS)}개 헤드')
    print(f'  -> 8개 헤드가 실질적으로 {eff_rank:.1f}개 몫만 하고 있다')

    print(f'\n  --- H3: +27점(100등)까지 필요한 ambiguity ---')
    need_bs = bs_ens - 27.0 / K
    print(f'  현재 ambiguity = {ambiguity:.6f}')
    print(f'  개별오차를 그대로 두고 ambiguity만으로 27점을 벌려면')
    print(f'    필요 ambiguity = {ambiguity + 27.0/K:.6f}  (현재의 {(ambiguity+27.0/K)/ambiguity:.2f}배)')
    print(f'  이는 헤드간 평균거리^2가 {(ambiguity+27.0/K)/ambiguity:.2f}배 커져야 한다는 뜻.')
    print(f'  (단, 다양성을 늘리면 개별오차도 같이 커지므로 실제로는 더 필요)')
