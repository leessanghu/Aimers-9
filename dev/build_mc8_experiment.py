"""mc8 = mc6의 wild를 판정축으로 재분할한 8분할 순수클래스 헤드.

mc6가 +9.77을 얻은 구조: 'nd&ball'처럼 뭉쳐있던 걸 성공/실패로 분리.
mc8은 같은 논리를 wild(공식실패2, 크게벗어남)에 적용:
  wild_ball 10.84%  그냥 크게 벗어난 볼
  wild_strk  1.85%  크게 벗어났는데 타자가 헛스윙/파울 -> '제구실패인데 스트라이크 획득'
  wild_play  0.47%  크게 벗어났는데 타자가 침
wild_strk는 물리적으로 완전히 다른 사건인데 mc6는 wild 하나로 뭉쳐놨다.

8분할 (전부 성공률 0% 또는 100%):
  0 middle    1 reverse   2 wild_ball  3 wild_strk  4 wild_play
  5 succ_ball 6 succ_strk 7 succ_play
  P(success) = P(5)+P(6)+P(7)

[대안] wild_play가 0.47%로 작아 학습 불안정 우려 -> mc7(wild_play를 wild_ball에 병합)도
동시 학습해서 비교한다.

로컬 rho는 반정보임이 확정됐으므로([[probe-first-methodology]]) 판정근거로 쓰지 않고,
mc6와의 d벡터 상관(독립성)만 본다. 실제 크기는 실측 프로브로 잰다.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
o = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[o[:-1]] = (pid[o][1:] == pid[o][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[o]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[o]] = np.nan
    lab = np.empty(n); lab[o] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
nd = valid & (mid < 0.5) & (rev < 0.5)

# --- mc8 (8분할) ---
cls8 = np.full(n, -1, np.int64)
cls8[valid & (mid > 0.5)] = 0
cls8[valid & (rev > 0.5) & (mid < 0.5)] = 1
wild = nd & (y == 0)
cls8[wild & (ball > 0.5)] = 2
cls8[wild & (strike > 0.5)] = 3
cls8[wild & (inplay > 0.5)] = 4
sc_ = nd & (y == 1)
cls8[sc_ & (ball > 0.5)] = 5
cls8[sc_ & (strike > 0.5)] = 6
cls8[sc_ & (inplay > 0.5)] = 7

# --- mc7 (wild_play를 wild_ball에 병합) ---
cls7 = cls8.copy()
cls7[cls8 == 4] = 2
cls7[cls8 == 5] = 4
cls7[cls8 == 6] = 5
cls7[cls8 == 7] = 6

CONFIGS = {
    'mc8': (cls8, 8, [5, 6, 7]),
    'mc7': (cls7, 7, [4, 5, 6]),
}
names8 = ['middle', 'reverse', 'wild_ball', 'wild_strk', 'wild_play',
          'succ_ball', 'succ_strk', 'succ_play']
print('=== mc8 클래스 분포 및 순수성 ===')
for c in range(8):
    m = cls8 == c
    print(f'  {c} {names8[c]:<11} n={m.sum():>9,} ({m.mean()*100:5.2f}%)  성공률={y[m].mean()*100:6.2f}%')
print(f'  미분류: {(cls8 < 0).sum():,} ({(cls8 < 0).mean()*100:.2f}%)')

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', early_stopping_rounds=50, random_seed=42)


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{m}.npy' for m in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{m}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{m}.npy') for m in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


# fold A에서만 학습해서 mc6와의 독립성(d상관) 확인 - 크기판정은 안 함
upto, vs = 2023, 2024
tr = (season <= upto)
va = season == vs
yv = y[va]
w = 0.5 ** ((upto - season[tr]) / 2.0)
H = build8('A')
W = {k: float(v95[f'{k}_weight']) for k in H}
t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
d_mc6 = np.load('dev/cache_mc6head_A.npy') - blend; d_mc6 -= d_mc6.mean()
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)

print(f'\n=== fold A 학습 (독립성 확인용, 크기판정 아님) ===')
for nm, (cls, ncls, succ_idx) in CONFIGS.items():
    trm = tr & (cls >= 0)
    wm = 0.5 ** ((upto - season[trm]) / 2.0)
    n_es = int(trm.sum() * 0.92)
    ts = time.time()
    m = CatBoostClassifier(**CB, classes_count=ncls)
    m.fit(X.loc[trm].iloc[:n_es], cls[trm][:n_es], sample_weight=wm[:n_es],
          eval_set=(X.loc[trm].iloc[n_es:], cls[trm][n_es:]))
    proba = m.predict_proba(X.loc[va])
    p = np.clip(proba[:, succ_idx].sum(axis=1), 0, 1)
    np.save(f'dev/cache_{nm}head_A.npy', p)
    d = p - blend; d -= d.mean()
    corr = float(np.mean(d * d_mc6) / np.sqrt(np.mean(d ** 2) * np.mean(d_mc6 ** 2)))
    A = float(np.mean(d * (blend - yv))); V = float(np.mean(d ** 2))
    print(f'  {nm}: BSS={sc(p):8.1f}  best_iter={m.best_iteration_}  '
          f'mc6와 d상관={corr:+.3f}  (로컬A={A:+.2e}, 참고용)  ({time.time()-ts:.0f}s)')
log('완료 - 독립성 확인됨. 프로덕션 학습은 별도 스크립트로.')
