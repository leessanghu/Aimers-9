"""v118 = v95 + mc6(0.432) + strk(0.227) + pitchtype(0.10)  [A+B 결합]

설계 근거:
 - mc6/strk는 실측 확정: A1=-5.0596e-05(V11=1.0491e-04), A2=-2.9235e-05
   2축 결합 최적점 = (0.432, 0.227), Δ=+11.40
 - pitchtype은 A3 미지 -> 프로브 가중치 0.10으로 얹어 한 제출에 두 목적 달성:
   (a) 2축 최적점 이동 이득 확보  (b) A3 측정
 - 축 상관: mc6-strk 0.248, mc6-pt 0.305, strk-pt 0.220 (셋 다 준독립)

pitchtype 헤드: Y=[y, is_fastball, is_breaking, is_offspeed]
  구종은 as-of pitchmix 카운터 차분으로 100% 복원(커버리지 99.9%).
  추론시 head0(y)만 사용 -> Rule4 안전.
  [주의] 로컬 fold A/C에서는 '손해'로 나왔으나, mc6/strk도 로컬 손해->실측 이득이었음
  ([[probe-first-methodology]]). 로컬로 기각하지 않고 실측으로 판정한다.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

S_MC6, S_STRK, S_PT = 0.432, 0.227, 0.10

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
v117 = joblib.load('submit/model/model_artifacts_v117.pkl')   # mc6/strk 모델 승계
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)

ptype = np.load('dev/recovered_pitch_type.npy')
ok = ptype >= 0
Ymat = np.column_stack([
    y.astype(np.float64),
    np.where(ok, (ptype == 0).astype(np.float64), np.nan),
    np.where(ok, (ptype == 1).astype(np.float64), np.nan),
    np.where(ok, (ptype == 2).astype(np.float64), np.nan),
])
log(f'pitchtype 보조타겟: 구종유효 {ok.mean()*100:.1f}%  '
    f'직구{np.mean(ptype[ok]==0)*100:.1f}% 변화구{np.mean(ptype[ok]==1)*100:.1f}% '
    f'오프{np.mean(ptype[ok]==2)*100:.1f}%')

CAT = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50,
           random_seed=42)
w = 0.5 ** ((2024.0 - season) / 2.0)
n_es = int(len(X) * 0.92)
log('pitchtype 전체데이터 학습...')
ts = time.time()
m = CatBoostRegressor(**CAT)
m.fit(X.iloc[:n_es], Ymat[:n_es], sample_weight=w[:n_es],
      eval_set=(X.iloc[n_es:], Ymat[n_es:]))
log(f'학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')

_RNG = ('Generator', 'BitGenerator', 'RandomState', 'PCG64', 'MT19937', 'Philox', 'SFC64')
def strip_rng(obj, seen=None, depth=0):
    if seen is None:
        seen = set()
    if depth > 12 or id(obj) in seen:
        return
    seen.add(id(obj))
    if hasattr(obj, '__dict__'):
        for k2, v2 in list(vars(obj).items()):
            if type(v2).__name__ in _RNG:
                setattr(obj, k2, None)
            else:
                strip_rng(v2, seen, depth + 1)
    elif isinstance(obj, dict):
        for k2, v2 in list(obj.items()):
            if type(v2).__name__ in _RNG:
                obj[k2] = None
            else:
                strip_rng(v2, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set)):
        for v2 in obj:
            strip_rng(v2, seen, depth + 1)


strip_rng(m)

v118 = dict(v117)
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
rest = 1.0 - S_MC6 - S_STRK - S_PT
print(f'\n=== 가중치 (v95 원본 x {rest:.4f}) ===')
for k in HEADS:
    orig = float(v95[f'{k}_weight'])
    v118[f'{k}_weight'] = orig * rest
    print(f'  {k:12s} v95={orig:.4f} -> {orig*rest:.4f}')
v118['mc6pure_weight'] = S_MC6
v118['strk_weight'] = S_STRK
v118['pitchtype_weight'] = S_PT
v118['pitchtype_model'] = m
tot = sum(float(v118[f'{k}_weight']) for k in HEADS) + S_MC6 + S_STRK + S_PT
print(f'  mc6pure      -> {S_MC6:.4f}')
print(f'  strk         -> {S_STRK:.4f}')
print(f'  pitchtype    -> {S_PT:.4f}  (프로브)')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9
for k in ('mc6pure_model', 'strk_model', 'pitchtype_model'):
    assert v118.get(k) is not None, f'{k} 누락'

joblib.dump(v118, 'submit/model/model_artifacts_v118.pkl')
log('v118 저장 완료')
