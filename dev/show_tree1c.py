import sys, json
sys.stdout.reconfigure(encoding='utf-8')
import joblib
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
feats = v88['feature_order']
with open('dev/tmp_tree.json', encoding='utf-8') as f:
    d = json.load(f)
ff = d['features_info']['float_features']
print('float_features 개수', len(ff))
print('예시 첫 항목:', ff[0])
idxs = [70, 25, 149, 1, 3]
for i in idxs:
    fi = ff[i]
    print(i, '->', fi)
