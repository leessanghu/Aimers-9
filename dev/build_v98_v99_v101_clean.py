"""단일축 설계 - 실측 나오면 바로 델타 분해 가능하도록.
예산 BUDGET=0.1727(=multires+condball+countresid 전량)을 비례로 빼서:
  v98: BUDGET의 30.12%(=0.052)만 mc5로 이전 (mc5: 13.8%->19.0%)
  v99: BUDGET 전량을 mc5로 이전 (mc5: 13.8%->31.1%)  [기존과 동일, 재확인용]
  v101: BUDGET 전량을 hurdle로 이전 (hurdle: 19.0%->36.3%) [mc5축과 동일예산 비교]
base/hurdle(v98,v99)/ordinal/midother/future50/ingame은 세 버전 모두 v95와 동일하게 유지.
"""
import joblib

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
keys = ['base_weight', 'hurdle_weight', 'multires_weight', 'ordinal_weight', 'midother_weight',
        'condball_weight', 'countresid_weight', 'future50_weight', 'mc5_weight', 'ingame_weight']

W0 = {k: v95[k] for k in keys}
DONORS = ['multires_weight', 'condball_weight', 'countresid_weight']
BUDGET = sum(W0[k] for k in DONORS)
print(f'BUDGET(도너 3개 합) = {BUDGET:.4f}')
frac = {k: W0[k] / BUDGET for k in DONORS}
print('도너별 비중:', frac)


def make(target_key, transfer_amount, tag):
    v = dict(v95)
    ratio = transfer_amount / BUDGET
    for k in DONORS:
        v[k] = W0[k] * (1 - ratio)
    v[target_key] += transfer_amount
    tot = sum(v[k] for k in keys)
    assert abs(tot - 1.0) < 1e-9, tot
    print(f'\n=== {tag} (이전량={transfer_amount:.4f}, {target_key}: {W0[target_key]:.4f}->{v[target_key]:.4f}) ===')
    for k in keys:
        mark = ' *' if v[k] != W0[k] else ''
        print(f'  {k:20s} {v[k]:.4f}{mark}')
    return v


v98 = make('mc5_weight', 0.052, 'v98 (mc5 부분이전)')
v99 = make('mc5_weight', BUDGET, 'v99 (mc5 전량이전)')
v101 = make('hurdle_weight', BUDGET, 'v101 (hurdle 전량이전, mc5축과 동일예산)')

joblib.dump(v98, 'submit/model/model_artifacts_v98.pkl')
joblib.dump(v99, 'submit/model/model_artifacts_v99.pkl')
joblib.dump(v101, 'submit/model/model_artifacts_v101.pkl')
print('\nv98/v99/v101 저장 완료')
