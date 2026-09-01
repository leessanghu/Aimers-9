import joblib
v = joblib.load('submit/model/model_artifacts_v117.pkl')
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
print('=== v117 헤드 가중치 ===')
tot = 0.0
for k in HEADS + ['mc6pure', 'strk']:
    w = float(v.get(f'{k}_weight', 0.0))
    tot += w
    print(f'  {k:<12} {w:.4f}')
print(f'  합계 = {tot:.6f}')
print(f'\n=== 후처리 보정 ===')
for k in ['risk_alpha', 'risk_thr', 'k2_alpha', 'k2_K', 'level_shift', 'w_hgb', 'w_cat']:
    print(f'  {k} = {v.get(k)}')
print(f'\n=== base 헤드 내부(HGB/CatBoost 변종) ===')
print(f'  hgbs: {len(v.get("hgbs", []))}개')
print(f'  cats: {len(v.get("cats", []))}개')
