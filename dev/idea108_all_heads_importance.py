import joblib
import numpy as np

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
feats = v88['feature_order']

heads = {}
heads['base(cat, w_cat={:.2f})'.format(v88['w_cat'])] = [np.array(m.get_feature_importance()) for m in v88['cats']]
heads['multires'] = [np.array(v88['multires_model'].get_feature_importance())]
heads['midother'] = [np.array(v88['midother_model'].get_feature_importance())]
heads['condball'] = [np.array(v88['condball_model'].get_feature_importance())]
heads['countresid'] = [np.array(v88['countresid_model'].get_feature_importance())]
heads['future50'] = [np.array(v88['future50_model'].get_feature_importance())]
heads['mc5'] = [np.array(v88['mc5_model'].get_feature_importance())]
heads['ingame'] = [np.array(v88['ingame_model'].get_feature_importance())]

weights = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
               ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
               countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
               ingame=v88['ingame_weight'])

print('=== 헤드별 상위 15개 피처 (CatBoost 계열만 - HGB인 base_hgb/hurdle/ordinal은 중요도 미제공) ===')
for name, imps in heads.items():
    avg = np.mean(imps, axis=0)
    order = np.argsort(avg)[::-1][:15]
    print(f'\n--- {name} ---')
    for i in order:
        print(f'  {feats[i]:35s} {avg[i]:.3f}')

print('\n\n=== season / cat_game_type 순위 (헤드별) ===')
si = feats.index('season'); gi = feats.index('cat_game_type')
for name, imps in heads.items():
    avg = np.mean(imps, axis=0)
    order = np.argsort(avg)[::-1]
    rank_s = list(order).index(si) + 1
    rank_g = list(order).index(gi) + 1
    print(f'  {name:15s} season=rank{rank_s:3d}(imp={avg[si]:.3f})   cat_game_type=rank{rank_g:3d}(imp={avg[gi]:.3f})')

print('\n\n=== 헤드 가중치(참고) ===')
for k, w in weights.items():
    print(f'  {k:12s} {w:.4f}')

print('\n\n=== 미제공(HGB 계열, 별도 permutation 필요) ===')
print('  base_hgb(w_hgb={:.2f}), hurdle(core_fail+succ_nc), ordinal_stage1/2/3'.format(v88['w_hgb']))
