"""v122 = v117 + xgbunused 헤드(음수가중치, w=-0.03) 프로브.
fold A/C 둘 다 s*<0으로 부호일치 확인됨(로컬 -0.049/-0.220, 크기는 안 믿음).
소량 음수가중치로 실측 프로브 -> A/V 역산용.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W_NEW = -0.03

v117 = joblib.load('submit/model/model_artifacts_v117.pkl')
v122 = dict(v117)

HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame', 'mc6pure', 'strk']
print('=== 가중치 재배분 (기존 전부 비례확대, 합계=1 유지) ===')
scale = 1 - W_NEW   # W_NEW가 음수라 scale>1 (다른 헤드 비중이 늘어남)
tot = 0.0
for k in HEADS:
    wk = f'{k}_weight'
    if wk not in v117:
        print(f'  [skip] {wk} 없음')
        continue
    old = float(v117[wk])
    new = old * scale
    v122[wk] = new
    tot += new
    print(f'  {k:12s} {old:.4f} -> {new:.4f}')

v122['xgbunused_weight'] = W_NEW
tot += W_NEW
print(f'  xgbunused    0.0000 -> {W_NEW:.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'가중치 합계 오류: {tot}'

xu = joblib.load('dev/xgbunused_production.pkl')
v122['xgbunused_model'] = xu['model']
v122['xgbunused_feat_order'] = xu['feat_order']
v122['xgbunused_raw_cols'] = xu['raw_cols']
v122['xgbunused_smap_season_tmm'] = xu['smap_season_tmm']
v122['xgbunused_smap_gtype_lown'] = xu['smap_gtype_lown']
v122['xgbunused_g_all'] = xu['g_all']
v122['xgbunused_k_shr'] = xu['k_shr']

joblib.dump(v122, 'submit/model/model_artifacts_v122.pkl')
print('\nv122 저장 완료: submit/model/model_artifacts_v122.pkl')
