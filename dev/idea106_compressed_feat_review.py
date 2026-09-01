import joblib

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
feats = v88['feature_order']
compressed = set([
    'asof_pitcher_success_rate_smooth', 'asof_pitcher_reverse_rate_smooth',
    'asof_pitcher_middle_rate_smooth', 'asof_pitcher_ball_rate_smooth',
    'asof_pitcher_strike_rate_smooth', 'asof_batter_success_rate_smooth',
    'asof_batter_middle_rate_smooth', 'asof_pitcher_fastball_rate_smooth',
    'asof_pitcher_breaking_rate_smooth', 'asof_pitcher_offspeed_rate_smooth',
    'inseason_success_smooth', 'inseason_ball_smooth', 'inseason_reverse_smooth',
    'bat_inseason_smooth', 'inseason_middle_smooth', 'inseason_strike_smooth',
    'pitcher_team_id_te', 'batter_team_id_te',
    'x_ability_here', 'x_count_pressure', 'x_ability_x_count', 'x_ability_x_pressure',
    'x_ability_x_inning', 'x_platoon_x_samehand', 'x_exp_x_ability', 'x_p_over_b',
    'x_ball_over_strike', 'x_rev_over_succ', 'x_mid_over_succ', 'x_kal_minus_career',
    'x_prev5_minus_career', 'x_prev1_minus_prev5', 'inseason_cmd_index',
])

model_keys = ['mc5_model', 'midother_model', 'condball_model', 'countresid_model', 'future50_model']
agg = {f: 0.0 for f in feats}
cnt = 0
for mk in model_keys:
    m = v88.get(mk)
    if m is None:
        continue
    try:
        imp = m.get_feature_importance()
    except Exception as e:
        print(mk, 'skip', e)
        continue
    cnt += 1
    for f, i in zip(feats, imp):
        agg[f] += i

print('사용모델수', cnt)
avgimp = {f: agg[f] / max(cnt, 1) for f in feats}
ranked = sorted(avgimp.items(), key=lambda kv: -kv[1])
rank_of = {f: i for i, (f, _) in enumerate(ranked)}

print()
print('=== 압축피처들의 평균순위/중요도 (전체 162개중) ===')
for f in sorted(compressed, key=lambda f: rank_of.get(f, 999)):
    print(f'  rank{rank_of.get(f, -1):3d}  imp={avgimp.get(f, 0):.3f}   {f}')

print()
print('=== 압축피처가 아닌 원본 피처 상위 20 (비교용) ===')
noncompressed = [f for f in feats if f not in compressed]
for f in sorted(noncompressed, key=lambda f: rank_of.get(f, 999))[:20]:
    print(f'  rank{rank_of.get(f, -1):3d}  imp={avgimp.get(f, 0):.3f}   {f}')
