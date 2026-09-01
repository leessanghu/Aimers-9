import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib, numpy as np

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
print(f'키 개수 {len(v95)}\n')
for k in sorted(v95.keys()):
    v = v95[k]
    t = type(v).__name__
    if isinstance(v, list):
        inner = type(v[0]).__name__ if v else '-'
        print(f'  {k:<34} list[{len(v)}] of {inner}')
    elif isinstance(v, (int, float, np.floating)):
        print(f'  {k:<34} {t} = {v}')
    elif isinstance(v, dict):
        print(f'  {k:<34} dict({len(v)}) keys={list(v.keys())[:6]}')
    elif isinstance(v, np.ndarray):
        print(f'  {k:<34} ndarray{v.shape}')
    else:
        print(f'  {k:<34} {t}')
