import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib, json

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
m = v88['condball_model']  # CatBoost 회귀헤드
feats = v88['feature_order']

m.save_model('dev/tmp_tree.json', format='json')
with open('dev/tmp_tree.json', encoding='utf-8') as f:
    d = json.load(f)

trees = d['oblivious_trees'] if 'oblivious_trees' in d else d.get('trees')
print(f'전체 트리 개수 = {len(trees)}')
t0 = trees[0]
print('\n=== 트리1(0번째) 구조 ===')
print(json.dumps(t0, indent=2, ensure_ascii=False)[:3000])

# 분기 피처 인덱스 -> 실제 이름
fkeys = d.get('features_info', {})
float_feats = fkeys.get('float_features', [])
print('\n=== 트리1이 쓰는 분기 피처(순서대로) ===')
if 'splits' in t0:
    for sp in t0['splits']:
        fidx = sp.get('float_feature_index')
        border = sp.get('border')
        if fidx is not None and fidx < len(float_feats):
            fname = float_feats[fidx].get('feature_id', f'f{fidx}')
            print(f'  분기: {fname}  <=  {border}')
