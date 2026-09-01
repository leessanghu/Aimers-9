import joblib
v121 = joblib.load('submit/model/model_artifacts_v121.pkl')
ALL = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
       'condball', 'countresid', 'future50', 'mc5', 'ingame',
       'mc6pure', 'strk']
tot = 0.0
for k in ALL:
    w = float(v121.get(f'{k}_weight', 0.0))
    tot += w
    print(f'  {k:12s} {w:.4f}')
print(f'진짜 합계(mc5/ingame 포함) = {tot:.6f}')
