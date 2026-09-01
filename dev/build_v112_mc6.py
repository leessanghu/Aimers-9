"""v112 = v95 + mc6원본 헤드(w=0.03).

mc6원본: 6-way 순수분할(middle/reverse/wild/succ_ball/succ_strk/succ_play) joint
MultiClass CatBoost. build_mc6_pure_head.py와 동일 레시피, 전체데이터(2019-2024)로
프로덕션 재학습.

[실험 배경] fold A 클린검증 -2.28(대조군 +1.69 미달) - 검증 미통과.
단, 오늘 실측한 v104/v107/v108 세 쌍(fold A local, 실측Δ)이 전부 "fold A 클수록
실측 더 나쁨" 패턴을 보였고, 이를 선형외삽하면 fold A가 음수인 후보는 실측이
오히려 양수일 수 있다는 가설(사용자 경험적 직관, n=3 매우 약한 회귀로 방향성만 확인).
w=0.03 선택근거: SE 기반 계산으로 이 가중치에서 이미 z~29 검출력 확보, 그 이상
올려도 검출력은 포화되고 손해(XGB선례 배율)만 커짐. 실험적 제출.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

W_NEW = 0.03

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
order = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(n); lab[order] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)

cls = np.full(n, -1, dtype=np.int64)
cls[valid & (mid > 0.5)] = 0
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y == 0)] = 2
cls[nd & (y == 1) & (ball > 0.5)] = 3
cls[nd & (y == 1) & (strike > 0.5)] = 4
cls[nd & (y == 1) & (inplay > 0.5)] = 5
SUCC_CLASSES = [3, 4, 5]
log(f'클래스 분포: ' + '  '.join(f'{c}:{(cls==c).mean()*100:.1f}%' for c in range(6))
    + f'  미분류:{(cls<0).mean()*100:.2f}%')

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=6, early_stopping_rounds=50,
          random_seed=42)

mask = cls >= 0
w = 0.5 ** ((2024.0 - season) / 2.0)
Xtr, ctr, wtr = X.loc[mask], cls[mask], w[mask]
n_es = int(len(Xtr) * 0.92)

log('mc6 전체데이터 학습...')
m = CatBoostClassifier(**CB)
m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=wtr[:n_es],
      eval_set=(Xtr.iloc[n_es:], ctr[n_es:]))
log(f'학습완료 best_iter={m.best_iteration_}')

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

v112 = dict(v95)
HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
         'condball', 'countresid', 'future50', 'mc5', 'ingame']
print('\n=== 가중치 재배분 (기존 전부 비례축소) ===')
for k in HEADS:
    old = float(v95[f'{k}_weight'])
    new = old * (1 - W_NEW)
    v112[f'{k}_weight'] = new
    print(f'  {k:12s} {old:.4f} -> {new:.4f}')
v112['mc6pure_weight'] = W_NEW
v112['mc6pure_model'] = m
v112['mc6pure_succ_classes'] = SUCC_CLASSES
tot = sum(float(v112[f'{k}_weight']) for k in HEADS) + W_NEW
print(f'  mc6pure      0.0000 -> {W_NEW:.4f}')
print(f'  합계 = {tot:.6f}')
assert abs(tot - 1.0) < 1e-9

joblib.dump(v112, 'submit/model/model_artifacts_v112.pkl')
log('v112 저장 완료')
