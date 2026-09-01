"""A1(mc6aux) 프로덕션: CatBoost MultiRMSEWithMissingValues([y, onehot6]), 전체데이터.

fold A 스크리닝(2026-08-31): rho=-0.00526, 클린H1/H2 +3.00(대조군 +1.70) 통과.
메커니즘: head0=y 직접 제곱오차회귀(Brier와 목적함수 일치) + mc6 6클래스를
        보조타겟으로 - 타겟재정의(mc6)와 멀티태스크(기존 6헤드 방식)의 결합.
방향: 음수(빼기, s* 로컬 -0.30이나 크기 불신). 추론은 head0만 -> Rule4 안전.

라벨/하이퍼파라미터는 fold 실험(build_mc6_family_A1A2A3.py A1)과 완전히 동일.
학습시간 3~4시간 예상.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.0f}s] {m}', flush=True)

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
log('클래스 분포: ' + '  '.join(f'{c}:{(cls==c).mean()*100:.1f}%' for c in range(6))
    + f'  미분류:{(cls<0).mean()*100:.2f}% (미분류행 onehot=NaN, y는 사용)')

# Y = [y, onehot(c0..c5)] - 미분류행은 onehot 전부 NaN (MultiRMSEWithMissingValues가 제외)
onehot = np.full((n, 6), np.nan)
okc = cls >= 0
for c in range(6):
    onehot[okc, c] = (cls[okc] == c).astype(np.float64)
Ymat = np.column_stack([y, onehot])

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50, random_seed=42)

w = 0.5 ** ((2024.0 - season) / 2.0)
n_es = int(n * 0.92)

log('전체데이터 학습 시작 (3~4시간 예상)...')
m = CatBoostRegressor(**CB)
m.fit(X.iloc[:n_es], Ymat[:n_es], sample_weight=w[:n_es],
      eval_set=(X.iloc[n_es:], Ymat[n_es:]))
log(f'학습완료 best_iter={m.best_iteration_}')

# 직렬화 왕복 검증
p_before = np.clip(m.predict(X.iloc[:1000])[:, 0], 0, 1)
joblib.dump(dict(model=m, feat_order=FEAT), 'dev/mc6aux_production.pkl')
rt = joblib.load('dev/mc6aux_production.pkl')
p_after = np.clip(rt['model'].predict(X.iloc[:1000][rt['feat_order']])[:, 0], 0, 1)
assert np.allclose(p_before, p_after, atol=1e-12), 'joblib 왕복 후 예측 불일치!'
log(f'직렬화 왕복 검증 통과 (최대차이 {np.abs(p_before-p_after).max():.2e})')
log('저장 완료: dev/mc6aux_production.pkl')
