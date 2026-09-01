import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib, numpy as np
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
feats = v88['feature_order']
model_keys = ['mc5_model', 'midother_model', 'condball_model', 'countresid_model', 'future50_model', 'ingame_model']
agg = np.zeros(len(feats))
cnt = 0
for mk in model_keys:
    m = v88.get(mk)
    if m is None:
        continue
    agg += np.array(m.get_feature_importance())
    cnt += 1
avgimp = agg / max(cnt, 1)
order = np.argsort(avgimp)
print('=== 하위 40개(6개 CatBoost 헤드 평균중요도, 낮은순) ===')
for rank, i in enumerate(order[:40]):
    print(f'  rank{rank+1:3d}  imp={avgimp[i]:.4f}   {feats[i]}')
