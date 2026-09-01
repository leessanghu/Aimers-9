import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
wkeys = sorted([k for k in v95 if k.endswith('_weight')])
print('=== v95의 모든 *_weight ===')
used, unused = [], []
for k in wkeys:
    v = v95[k]
    try:
        fv = float(v)
    except Exception:
        continue
    (used if fv > 0 else unused).append((k, fv))
print('  [사용중]')
for k, v in sorted(used, key=lambda x: -x[1]):
    print(f'    {k:28s} {v:.4f}')
print('  [가중치 0 = 만들어졌지만 미사용]')
for k, v in unused:
    base = k.replace('_weight', '')
    has_model = any(kk.startswith(base) and kk != k for kk in v95)
    print(f'    {k:28s} {v:.4f}   아티팩트존재={has_model}')

print()
print('=== 모델 객체는 있는데 쓰이는지 불명확한 키 ===')
for k in sorted(v95):
    if k.endswith('_model') or k.endswith('_models'):
        wk = k.replace('_models', '_weight').replace('_model', '_weight')
        w = v95.get(wk, None)
        print(f'  {k:28s} -> {wk}={w}')
