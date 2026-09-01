"""build_v105_honest_oof_isotonic.py가 밤새 얼마나 진행됐든, 그 시점까지 체크포인트된
OOF로 isotonic 맵을 만들고 v105 아티팩트를 조립한다. 5-fold 전부 안 끝난 헤드도
가용한 fold만으로 isotonic 학습(coverage가 충분하면). 아침에 이 스크립트만 실행."""
import sys, json, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.isotonic import IsotonicRegression

CKPT_DIR = 'dev/v105_ckpt'
PROGRESS_FILE = f'{CKPT_DIR}/progress.json'
MIN_COVERAGE = 0.5  # 최소 50% 이상 fold가 채워져야 그 헤드에 isotonic 적용(아니면 원본 유지)

# ★ v104 실패(cross-model-calibration-mismatch) 재발 방지 화이트리스트.
# build_v105는 시간상 일부 헤드를 단순화했다(base=단일CatBoost vs 프로덕션 HGB3+Cat3,
# hurdle=CatBoost 2단 vs 프로덕션 HGB 2단, multires/countresid/future50/ingame=
# y단순회귀 vs 프로덕션 multi-task). 이런 헤드는 OOF 출력분포가 프로덕션과 달라서
# isotonic 맵을 이식하면 v104와 똑같은 미스매치가 난다.
# 레시피가 프로덕션과 충실히 일치하는 헤드에만 적용한다.
FAITHFUL_HEADS = {'mc5', 'midother', 'condball', 'ordinal'}

meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
n = len(y)

progress = {}
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE) as f:
        progress = json.load(f)

print('=== 헤드별 진행상황 ===')
HEAD_ORDER = ['hurdle', 'mc5', 'base', 'midother', 'ordinal', 'ingame',
              'condball', 'countresid', 'future50', 'multires']
iso_maps = {}
for head in HEAD_ORDER:
    p = f'{CKPT_DIR}/oof_{head}.npy'
    done = progress.get(head, [])
    if not os.path.exists(p):
        print(f'  {head:12s} 미시작')
        continue
    oof = np.load(p)
    m = ~np.isnan(oof)
    cov = m.mean()
    print(f'  {head:12s} fold완료={len(done)}/5  coverage={cov*100:.1f}%  n={m.sum():,}')
    if head not in FAITHFUL_HEADS:
        print(f'    -> 레시피가 프로덕션과 불일치(단순화됨), isotonic 스킵 - v104 재발 방지')
        continue
    if cov < MIN_COVERAGE or m.sum() < 5000:
        print(f'    -> coverage 부족, isotonic 스킵(원본 가중치만 사용)')
        continue
    iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
    iso.fit(oof[m], y[m])
    iso_maps[head] = iso
    print(f'    -> isotonic map 학습 완료')

if not iso_maps:
    print('\n적용 가능한 헤드가 하나도 없음 - v105 생성 중단')
    sys.exit(1)

joblib.dump(iso_maps, f'{CKPT_DIR}/iso_maps_honest.pkl')
print(f'\niso_maps 저장: {len(iso_maps)}개 헤드 ({list(iso_maps.keys())})')

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v105 = dict(v95)
v105['iso_maps'] = iso_maps
joblib.dump(v105, 'submit/model/model_artifacts_v105.pkl')
print('v105 아티팩트 저장 완료: submit/model/model_artifacts_v105.pkl')
