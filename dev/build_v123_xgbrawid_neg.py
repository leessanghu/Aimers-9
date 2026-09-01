"""v123 = v122 + xgb_rawid 헤드(음수가중치 w=-0.03) 프로브.

근거:
 - v108(w=+0.03, 양수)에서 실측 -1.19 -> 항등식 역산 C=+2.7e-05(양수) = 최적방향은 음수.
   즉 음수가중치는 한 번도 실측한 적 없는 미탐색 방향.
 - v122(xgbunused 반영) 기준 fold A에서 xgbunused와 직교화 후에도 rho=-0.00579 (z=2.9)로
   신호가 거의 안 줄어듦 -> xgbunused와 잔차정렬 성분이 독립.
 - 대조군 검증 통과한 fold A 기준. fold C는 전역편향으로 정보 없음(오늘 확인).

가중치 -0.03 선택: 로컬 s*=-0.131이지만 로컬은 크기를 4~15배 과대추정하므로 보수적으로.
최악의 경우 손실 ~-0.07점, 기대이득 +0.1~+0.25점.
xgb_rawid 프로덕션 모델/카테고리는 v108에서 그대로 가져옴(재학습 불필요).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import joblib

W_NEW = -0.03

v122 = joblib.load('submit/model/model_artifacts_v122.pkl')
v108 = joblib.load('submit/model/model_artifacts_v108.pkl')
v123 = dict(v122)

HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame',
         'mc6pure', 'strk', 'xgbunused']
print('=== 가중치 재배분 (기존 전부 비례확대, 합계=1 유지) ===')
scale = 1 - W_NEW
tot = 0.0
for k in HEADS:
    wk = f'{k}_weight'
    if wk not in v122:
        print(f'  [skip] {wk} 없음')
        continue
    old = float(v122[wk])
    new = old * scale
    v123[wk] = new
    tot += new
    print(f'  {k:12s} {old:+.4f} -> {new:+.4f}')

v123['xgbrawid_weight'] = W_NEW
v123['xgbrawid_model'] = v108['xgbrawid_model']
v123['xgbrawid_cats'] = v108['xgbrawid_cats']
tot += W_NEW
print(f'  xgbrawid     +0.0000 -> {W_NEW:+.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9, f'가중치 합계 오류: {tot}'

joblib.dump(v123, 'submit/model/model_artifacts_v123.pkl')
print('\nv123 저장 완료: submit/model/model_artifacts_v123.pkl')
