import joblib
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
feats = v95['feature_order']
print(len(feats))
for f in feats:
    print(f)
