"""mc10을 '8헤드 블렌드'가 아니라 '실제 v117 전체 블렌드(mc6+strk 포함)' 기준으로 재평가.

기존 계산의 문제: rho/A/클린max-gain 전부 mc6/strk 넣기 전 8헤드 블렌드로 쟀음.
그런데 mc6와 상관 0.61~0.90이니, mc10 신호의 상당부분이 '이미 v117에 들어있는 mc6분'을
재발견한 것일 수 있음. v117 전체를 기준선으로 다시 재면 진짜 잔여가치가 나온다.

추가로: mc6 하나가 아니라 '현재 블렌드를 구성하는 모든 개별 요소'와의 상관도 같이 본다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_
NEED_RHO = 0.01740

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
S_MC6, S_STRK = 0.48, 0.10


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
    W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
    t0 = sum(W0.values()); W0 = {k: v / t0 for k, v in W0.items()}
    blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)   # 옛 8헤드 블렌드(참고용)

    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    blend_v117 = np.clip(rest * blend8 + S_MC6 * p_mc6 + S_STRK * p_strk, 0, 1)   # 진짜 v117

    p_mc10 = np.load(f'dev/cache_mc10head_{tag}.npy')

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
    print(f'\n{"="*78}\n=== fold {tag} ({vs}) ===\n{"="*78}')
    print(f'  8헤드만 BSS   = {sc(blend8):.1f}')
    print(f'  v117 전체 BSS = {sc(blend_v117):.1f}')

    for label, base_blend in [('8헤드 기준(예전 계산)', blend8), ('v117 전체 기준(올바른 계산)', blend_v117)]:
        resid = yv - base_blend
        d = p_mc10 - base_blend; dc = d - d.mean()
        V = float(np.mean(dc ** 2)); A = float(np.mean(dc * (base_blend - yv)))
        rho = -A / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
        gain = K * A ** 2 / V if V > 1e-14 else 0.0
        print(f'\n  [{label}]')
        print(f'    rho = {rho:+.5f} (필요치의 {abs(rho)/NEED_RHO*100:5.1f}%)  최대이득(로컬) = {gain:+.2f}점')

    # 개별 헤드 전체(8헤드+mc6+strk)와의 상관 - 전체 그림
    print(f'\n  --- mc10과 각 헤드의 d상관(8헤드블렌드 기준 편차) ---')
    d_mc10 = p_mc10 - blend8; d_mc10 -= d_mc10.mean()
    for k in HEADS8 + ['mc6', 'strk']:
        if k == 'mc6':
            hk = p_mc6
        elif k == 'strk':
            hk = p_strk
        else:
            hk = H[k]
        dk = hk - blend8; dk = dk - dk.mean()
        corr = float(np.mean(d_mc10 * dk) / np.sqrt(np.mean(d_mc10**2) * np.mean(dk**2) + 1e-18))
        print(f'    {k:<12} corr={corr:+.4f}')
