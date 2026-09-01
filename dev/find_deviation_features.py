import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib, re, numpy as np
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
order = np.argsort(avgimp)[::-1]
rank_of = {feats[i]: list(order).index(i) + 1 for i in range(len(feats))}

pat = re.compile(r'minus_career|diff_|_minus_|prev1_minus|prev5_minus|kal_minus|drift|trend|momentum|form', re.I)
cands = [f for f in feats if pat.search(f)]
print(f'=== "편차/트렌드" 계열 피처 {len(cands)}개 (전체 162 중 rank순) ===')
for f in sorted(cands, key=lambda f: rank_of[f]):
    print(f'  rank{rank_of[f]:3d}  {f}')
