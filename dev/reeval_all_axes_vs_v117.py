"""오늘/최근 스크리닝했던 모든 후보축을, 8헤드 블렌드가 아니라
실제 v117 전체 블렌드(mc6=0.48+strk=0.10+8헤드=0.42) 기준으로 재평가.
fold A/C 둘 다. XGB 계열도 재확인차 포함(이미 닫힌 걸로 결론났던 축).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib, os

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

CANDS = {
    'mc6h_wild':     'dev/cache_mc6h_headA_wild_{tag}.npy',
    'mc6h_ball':     'dev/cache_mc6h_headB_ball_{tag}.npy',
    'mc6h_strike':   'dev/cache_mc6h_headC_strike_{tag}.npy',
    'seqA_prev_y':   'dev/cache_seq_seqA_prev_y_{tag}.npy',
    'seqB_streak':   'dev/cache_seq_seqB_streak_{tag}.npy',
    'seqC_prevball': 'dev/cache_seq_seqC_prev_ball_{tag}.npy',
    'pitchtype':     'dev/cache_pitchtypehead_{tag}.npy',
    'persona':       'dev/cache_persona_{tag}.npy',
    'xgb_rawid':     'dev/cache_xgbrawid_{tag}.npy',
    'xgb_ctx':       'dev/cache_xgbctx_{tag}.npy',
    'xgb_hurdlectx': 'dev/cache_xgbhurdlectx_{tag}.npy',
    'lgbm_rawid':    'dev/cache_lgbmrawid_{tag}.npy',
}


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


results = {}
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    H = build8(tag)
    W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
    t0 = sum(W0.values()); W0 = {k: v / t0 for k, v in W0.items()}
    blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)

    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    blend_v117 = np.clip(rest * blend8 + S_MC6 * p_mc6 + S_STRK * p_strk, 0, 1)
    resid117 = yv - blend_v117
    E_resid117 = float(np.mean(resid117 ** 2))

    d_mc6 = p_mc6 - blend_v117; d_mc6 -= d_mc6.mean()
    d_strk = p_strk - blend_v117; d_strk -= d_strk.mean()

    print(f'\n{"="*90}\n=== fold {tag} ({vs}) — v117 전체 블렌드 기준 ===\n{"="*90}')
    print(f'{"후보":<16}{"rho(vs v117)":>14}{"필요치%":>9}{"로컬maxgain":>13}{"corr_mc6":>10}{"corr_strk":>10}')
    row_out = {}
    for nm, tmpl in CANDS.items():
        path = tmpl.format(tag=tag)
        if not os.path.exists(path):
            continue
        p = np.load(path)
        d = p - blend_v117; dc = d - d.mean()
        V = float(np.mean(dc ** 2))
        if V < 1e-14:
            continue
        A = float(np.mean(dc * (blend_v117 - yv)))
        rho = -A / np.sqrt(V * E_resid117)
        gain = K * A ** 2 / V
        corr_mc6 = float(np.mean(dc * d_mc6) / np.sqrt(V * np.mean(d_mc6**2) + 1e-18))
        corr_strk = float(np.mean(dc * d_strk) / np.sqrt(V * np.mean(d_strk**2) + 1e-18))
        row_out[nm] = (rho, gain, corr_mc6, corr_strk)
        print(f'{nm:<16}{rho:>+14.5f}{abs(rho)/NEED_RHO*100:>8.1f}%{gain:>+13.2f}{corr_mc6:>+10.3f}{corr_strk:>+10.3f}')
    results[tag] = row_out

print(f'\n{"="*90}\n=== 요약: fold A/C 모두에서 부호 일치 + rho 필요치대비 의미있는 크기 있는 후보 ===\n{"="*90}')
for nm in CANDS:
    if nm in results.get('A', {}) and nm in results.get('C', {}):
        rA, gA, cmA, csA = results['A'][nm]
        rC, gC, cmC, csC = results['C'][nm]
        agree = '동일부호' if rA * rC > 0 else '부호반전'
        print(f'  {nm:<16} A: rho={rA:+.5f} gain={gA:+6.2f}  |  C: rho={rC:+.5f} gain={gC:+6.2f}  -> {agree}')
