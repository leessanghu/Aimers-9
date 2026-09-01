"""XGB/LGBM을 메타블렌드가 아니라 'base 헤드 자체의 배깅 멤버'로 추가하면 도움되는지 테스트.
base = 현재 HGB 3변종(d6,d8,sub) 평균. 여기에 XGB/LGBM을 추가배깅했을 때
base 헤드 자체의 단독 BSS가 오르는지가 핵심(메타블렌드 레벨이 아니라 헤드 내부 레벨).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

B_ = 0.249807
K = 1e5 / B_
NEED_RHO = 0.01740
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)

for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    hgb3 = [np.load(f'dev/phase90_cache/{tag}_base_{n}.npy') for n in ('d6', 'd8', 'sub')]
    base = np.mean(hgb3, axis=0)
    p_xgb = np.load(f'dev/cache_xgbrawid_{tag}.npy')
    p_lgbm = np.load(f'dev/cache_lgbmrawid_{tag}.npy')

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
    print(f'\n{"="*78}\n=== fold {tag} ({vs}) ===\n{"="*78}')
    print(f'  base(HGB 3변종 평균) 단독 BSS = {sc(base):.2f}')
    for nm in ('d6', 'd8', 'sub'):
        print(f'    HGB_{nm} 단독 BSS = {sc(np.load(f"dev/phase90_cache/{tag}_base_{nm}.npy")):.2f}')
    print(f'  xgb_rawid 단독 BSS  = {sc(p_xgb):.2f}')
    print(f'  lgbm_rawid 단독 BSS = {sc(p_lgbm):.2f}')

    # HGB 3변종끼리의 상관(현재 다양성) vs XGB/LGBM과의 상관
    print(f'\n  --- HGB 멤버간 vs XGB/LGBM과의 예측상관(pairwise) ---')
    all_m = dict(d6=hgb3[0], d8=hgb3[1], sub=hgb3[2], xgb=p_xgb, lgbm=p_lgbm)
    names = list(all_m.keys())
    for i in range(len(names)):
        for j in range(i+1, len(names)):
            a, b = all_m[names[i]], all_m[names[j]]
            c = np.corrcoef(a, b)[0, 1]
            print(f'    {names[i]:<6} vs {names[j]:<6}  corr={c:.4f}')

    # base(현재 3배깅) 잔차 대비 XGB/LGBM 배깅추가 시 이득(honest, 원래 blend 안에 넣는 형태로)
    resid = yv - base
    E_r2 = float(np.mean(resid ** 2))
    print(f'\n  --- base 잔차 대비 XGB/LGBM 잔차상관(배깅멤버로 추가시 헤드룸) ---')
    for nm, p in [('xgb_rawid', p_xgb), ('lgbm_rawid', p_lgbm)]:
        d = p - base; dc = d - d.mean()
        V = float(np.mean(dc ** 2)); A = float(np.mean(dc * (base - yv)))
        rho = -A / np.sqrt(V * E_r2) if V > 1e-14 else 0.0
        gain = K * A ** 2 / V if V > 1e-14 else 0.0
        print(f'    {nm:<12} rho={rho:+.5f} ({abs(rho)/NEED_RHO*100:5.1f}%)  로컬최대이득={gain:+.2f}')

    # 실제 배깅(단순평균 4번째/5번째 멤버로 추가)했을 때 base 자체의 BSS 변화 - 직접 측정
    print(f'\n  --- 실제 단순배깅(동일가중 평균) 시 base 단독 BSS 변화 ---')
    combos = {
        'base(HGB3만, 현재)': base,
        '+xgb 25%': 0.75*base + 0.25*p_xgb,
        '+xgb 50%': 0.50*base + 0.50*p_xgb,
        '+lgbm 25%': 0.75*base + 0.25*p_lgbm,
        '+lgbm 50%': 0.50*base + 0.50*p_lgbm,
        'HGB3+xgb+lgbm 균등(5멤버)': np.mean(hgb3 + [p_xgb, p_lgbm], axis=0),
        'HGB3+xgb+lgbm 균등(4멤버,d6d8만+xgb+lgbm)': np.mean([hgb3[0], hgb3[1], p_xgb, p_lgbm], axis=0),
    }
    for nm, pp in combos.items():
        print(f'    {nm:<38} BSS={sc(pp):8.2f}  (vs base 현재: {sc(pp)-sc(base):+7.2f})')
