"""F전문가 속성 빌드: F행(11%)만으로 mc6 학습 - 기존 mc6의 F행 예측만 교체하는 용도.

검증: fold A(2024)의 F행에서 [F전문가 vs 기존 공유 mc6] 직접 Brier 비교.
      (잔차상관 스크리닝이 아니라 같은 홀드아웃에서의 직접 성능비교 - 신뢰도 높음)
통과시: 전체데이터 F전문가 학습 + 저장. 전부 ~25-30분.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807

X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y_all = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95a['feature_order'])
X = X_df[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
test_gt = pd.read_csv('data/test.csv', encoding='utf-8-sig', usecols=['game_type'])
r_value = test_gt['game_type'].iloc[0]
is_R = (df['game_type'] == r_value).to_numpy()
is_F = ~is_R
print(f'F행 비율 {is_F.mean()*100:.1f}%')

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


rev = diff_label('asof_pitcher_reverse_rate'); mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
cls = np.full(n, -1, dtype=np.int64)
cls[valid & (mid > 0.5)] = 0
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y_all == 0)] = 2
cls[nd & (y_all == 1) & (ball > 0.5)] = 3
cls[nd & (y_all == 1) & (strike > 0.5)] = 4
cls[nd & (y_all == 1) & (inplay > 0.5)] = 5
SUCC = [3, 4, 5]

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=6, early_stopping_rounds=50,
          random_seed=42)

# ---- fold A F전문가 ----
tr = (season <= 2023) & (cls >= 0) & is_F
va = season == 2024
vaF = va & is_F
yvF = y_all[vaF]
w = 0.5 ** ((2023.0 - season[tr]) / 2.0)
Xtr, ctr = X.loc[tr], cls[tr]
n_es = int(len(Xtr) * 0.92)
log(f'F전문가 fold A 학습행 {tr.sum():,}, 검증 F행 {vaF.sum():,}')
m = CatBoostClassifier(**CB)
m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=w[:n_es],
      eval_set=(Xtr.iloc[n_es:], ctr[n_es:]))
pF = np.clip(m.predict_proba(X.loc[vaF])[:, SUCC].sum(axis=1), 0, 1)
log(f'학습완료 best_iter={m.best_iteration_}')

# 기존 mc6의 F행 예측과 직접 비교
p_mc6_all = np.load('dev/cache_mc6head_A.npy')      # fold A 전체(2024) 예측
maskF_in_va = is_F[va]
p_mc6_F = p_mc6_all[maskF_in_va]
bs = lambda pp: float(np.mean((np.clip(pp, 0, 1) - yvF) ** 2))
bss = lambda pp: 1e5 * (1 - bs(pp) / B_)
print(f'\n=== fold A F행({vaF.sum():,}개) 직접 비교 ===')
print(f'  F전문가   Brier={bs(pF):.6f}  BSS={bss(pF):8.2f}')
print(f'  공유 mc6  Brier={bs(p_mc6_F):.6f}  BSS={bss(p_mc6_F):8.2f}')
diff = bss(pF) - bss(p_mc6_F)
print(f'  차이 = {diff:+.2f} (양수면 F전문가 승)')
# 전체 test에 미치는 영향 추정: F행 비중 x (mc6 가중치 0.44) x 차이
print(f'  참고: 2025 F비중이 fold A와 같다면({maskF_in_va.mean()*100:.1f}%), '
      f'블렌드 이득 대략 {diff * maskF_in_va.mean() * 0.44:+.2f}점')

if diff <= 0:
    log('F전문가가 공유 mc6보다 못함 - 중단')
    sys.exit(0)

# ---- 프로덕션 F전문가 (전체데이터 F행) ----
log('프로덕션 F전문가 학습...')
trP = (cls >= 0) & is_F
wP = 0.5 ** ((2024.0 - season[trP]) / 2.0)
XtrP, ctrP = X.loc[trP], cls[trP]
n_esP = int(len(XtrP) * 0.92)
mP = CatBoostClassifier(**CB)
mP.fit(XtrP.iloc[:n_esP], ctrP[:n_esP], sample_weight=wP[:n_esP],
       eval_set=(XtrP.iloc[n_esP:], ctrP[n_esP:]))
log(f'프로덕션 완료 best_iter={mP.best_iteration_}')
joblib.dump(dict(model=mP, feat_order=FEAT, succ_classes=SUCC, r_value=r_value),
            'dev/fexpert_production.pkl')
log('저장: dev/fexpert_production.pkl')
