"""mc5 디코더 업그레이드: E[y|class] (11 상수) -> E[y|class, count] (11 x count).
현재 프로덕션은 count 정보를 디코딩 단계에서 버린다.
이건 새 헤드 추가가 아니라 '가장 성공한 축(mc5, +10.5)'의 디코더 교체.
Rule4 안전: E[y|class,count] 는 train<=2023 통계로만 계산, 각 행은 자기 count만 참조.
fold A H1<->H2 정직 검증.
"""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('data/train.csv', encoding='utf-8-sig',
                 usecols=['row_id', 'season', 'balls_before', 'strikes_before', 'control_success'])
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
df = df.sort_values('row_num').reset_index(drop=True)
df['count_state'] = df['balls_before'] * 3 + df['strikes_before']  # 0..11

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807

# 11-class 라벨 복원 (idea82와 동일 방식)
CLS5 = np.load('dev/cls5_labels.npy')
PT = np.load('dev/pitchtype_labels.npy')
valid11 = (CLS5 >= 0) & (PT >= 0)
CLS11 = np.full(len(CLS5), -1, dtype=np.int64)
nd = valid11 & (CLS5 >= 2)
CLS11[nd] = (CLS5[nd] - 2) * 3 + PT[nd]
CLS11[valid11 & (CLS5 == 0)] = 9
CLS11[valid11 & (CLS5 == 1)] = 10

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
p = 'A'
H = dict(
    base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/{p}_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/{p}_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42, 7)]),
)
P11 = np.load('dev/idea75_cache/A_proba11.npy')
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)

# ---- 디코더 두 종류를 train<=2023 에서 학습 ----
tr = season <= 2023
cls_tr = CLS11[tr]; y_tr = y[tr]; cnt_tr = df.loc[tr, 'count_state'].to_numpy()
succ_const = np.array([y_tr[cls_tr == c].mean() if (cls_tr == c).sum() > 0 else y_tr.mean()
                       for c in range(11)])
print('현행 디코더 E[y|class] =', np.round(succ_const, 4))
print()

# count-conditioned, 소표본 축소 (K=200, 전역 class 평균으로 shrink)
K = 200.0
succ_bycount = np.zeros((11, 12))
for c in range(11):
    for k in range(12):
        m = (cls_tr == c) & (cnt_tr == k)
        n = m.sum()
        if n > 0:
            succ_bycount[c, k] = (y_tr[m].sum() + K * succ_const[c]) / (n + K)
        else:
            succ_bycount[c, k] = succ_const[c]
print('count별 변동폭이 큰 class (max-min):')
for c in range(11):
    rng = succ_bycount[c].max() - succ_bycount[c].min()
    print(f'  class {c:2d}: 상수={succ_const[c]:.4f}  count별 범위={rng:.4f}  [{succ_bycount[c].min():.3f}, {succ_bycount[c].max():.3f}]')
print()

cnt_va = df.loc[va, 'count_state'].to_numpy()
mc5_const = np.clip(P11 @ succ_const, 0, 1)
dec_rows = succ_bycount[:, cnt_va]          # (11, n_va)
mc5_count = np.clip((P11.T * dec_rows).sum(axis=0), 0, 1)

sc = lambda q, m: 1e5 * (1 - np.mean((np.clip(q[m], 0, 1) - yv[m]) ** 2) / unc)
allm = np.ones(len(yv), bool)
print(f'mc5 단독: 상수디코더={sc(mc5_const, allm):.2f}   count디코더={sc(mc5_count, allm):.2f}  ({sc(mc5_count,allm)-sc(mc5_const,allm):+.2f})')
print()

risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))


def full(mc5_pred):
    H2 = dict(H); H2['mc5'] = mc5_pred
    raw = sum(W[k] * H2[k] for k in H2)
    return np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)


f_const = full(mc5_const)
f_count = full(mc5_count)
print('=== v88 전체 블렌드 기준 ===')
print(f'  상수디코더(현행) = {sc(f_const, allm):.2f}')
print(f'  count디코더      = {sc(f_count, allm):.2f}   ({sc(f_count,allm)-sc(f_const,allm):+.2f})')
print()

H1 = mth <= 6; H2m = ~H1
print('=== H1<->H2 정직검증 (디코더는 train<=2023 고정, 데이터 분할과 무관) ===')
for tag, m in [('H1', H1), ('H2', H2m)]:
    print(f'  {tag}: 상수={sc(f_const, m):.2f}  count={sc(f_count, m):.2f}  ({sc(f_count,m)-sc(f_const,m):+.2f})')
