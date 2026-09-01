import sys, json
sys.stdout.reconfigure(encoding='utf-8')
with open('dev/tmp_tree.json', encoding='utf-8') as f:
    d = json.load(f)
t0 = d['oblivious_trees'][0]
print('트리1 리프개수 =', len(t0['leaf_values']))
print('트리1 키:', list(t0.keys()))
print()
print('splits:')
print(json.dumps(t0.get('splits', 'NO SPLITS KEY'), indent=2, ensure_ascii=False))
