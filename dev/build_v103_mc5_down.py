"""v103 = v95에서 mc5를 오히려 낮춤(v98과 반대방향, 동일 크기).
mc5: 13.8% -> 8.6% (-0.052), 뺀 만큼 donor 3개(multires/condball/countresid)에 비례로 돌려줌.
base/hurdle/기타는 v95와 동일 유지(클린 단일축)."""
import joblib

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v103 = dict(v95)

keys = ['base_weight', 'hurdle_weight', 'multires_weight', 'ordinal_weight', 'midother_weight',
        'condball_weight', 'countresid_weight', 'future50_weight', 'mc5_weight', 'ingame_weight']
DONORS = ['multires_weight', 'condball_weight', 'countresid_weight']
W0 = {k: v95[k] for k in keys}
DONOR_BUDGET = sum(W0[k] for k in DONORS)

MOVE = 0.052
frac = {k: W0[k] / DONOR_BUDGET for k in DONORS}
for k in DONORS:
    v103[k] = W0[k] + MOVE * frac[k]
v103['mc5_weight'] = W0['mc5_weight'] - MOVE

print('=== v103 (mc5 하향, v98과 반대방향 동일크기) ===')
tot = 0.0
for k in keys:
    mark = ' *' if v103[k] != W0[k] else ''
    print(f'  {k:20s} {W0[k]:.4f} -> {v103[k]:.4f}{mark}')
    tot += v103[k]
print(f'합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v103, 'submit/model/model_artifacts_v103.pkl')
print('\nv103 저장 완료')
