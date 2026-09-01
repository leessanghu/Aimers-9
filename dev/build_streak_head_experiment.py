"""연속실패 축 제대로 인코딩 + seqC_prev_ball 결합 실험.

포렌식 발견(raw_forensics_v3):
  0연속실패 0.5434 / 1연속 0.5197 / 2연속 0.5000 / 3연속 0.4831 / 4연속 0.4700 / 5+연속 0.4474
  => 0 vs 5+ 격차 9.6%p, 완벽한 단조. 오늘 본 어떤 축보다 큼.
이전 seq 실험에서 clip(0,3)/3 으로 잘라 4연속/5+연속 신호를 날렸음 -> 재실험.

인코딩 3종 비교:
  vA: 원값 clip(0,10) / 10        (선형, 꼬리 보존)
  vB: log1p(streak) / log1p(10)   (앞쪽 구간 해상도 ↑)
  vC: '5+연속실패인가' 이진        (꼬리만 집중)
+ seqC_prev_ball(직전 실험 최강, foldA rho -0.00399)과의 2타겟 결합도 테스트.

전부 [y, aux...] multi-task. 추론은 head0만 -> Rule4 안전.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807
K = 1e5 / B
NEED_RHO = 0.01740

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

same_prev = np.zeros(len(o), dtype=bool)
same_prev[1:] = (pid[o][1:] == pid[o][:-1])
dn_ord = np.full(len(o), np.nan)
dn_ord[1:] = n_[o][1:] - n_[o][:-1]
valid_prev_ord = same_prev & (dn_ord == 1)

# 연속실패 (clip 없이 원값)
yo = y[o]
streak_ord = np.zeros(len(o))
cur = 0.0
for i in range(len(o)):
    if not valid_prev_ord[i]:
        cur = 0.0
    streak_ord[i] = cur
    cur = 0.0 if yo[i] == 1 else cur + 1
streak_raw = np.empty(n); streak_raw[o] = streak_ord
okm = np.full(n, False); okm[o] = valid_prev_ord
streak_raw = np.where(okm, streak_raw, np.nan)

ball = call[:, 0]
prev_ball_ord = np.full(len(o), np.nan); prev_ball_ord[1:] = ball[o][:-1]
prev_ball = np.full(n, np.nan); prev_ball[o] = np.where(valid_prev_ord, prev_ball_ord, np.nan)

sA = np.where(okm, np.clip(streak_raw, 0, 10) / 10.0, np.nan)
sB = np.where(okm, np.log1p(np.clip(streak_raw, 0, 10)) / np.log1p(10), np.nan)
sC = np.where(okm, (streak_raw >= 5).astype(np.float64), np.nan)

print('연속실패 분포 확인:')
v = streak_raw[okm]
for k in range(6):
    m = v == k
    print(f'  {k}연속: {m.sum():>9,} ({m.mean()*100:5.2f}%)')
print(f'  5+연속: {(v>=5).sum():>9,} ({(v>=5).mean()*100:5.2f}%)  최대={v.max():.0f}')

CONFIGS = {
    'strk_linear':   [sA],
    'strk_log':      [sB],
    'strk_tail5':    [sC],
    'strk_log+ball': [sB, prev_ball],
    'strk_all3':     [sA, sB, sC],
}

CAT = dict(iterations=1200, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50,
           random_seed=42)


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


results = {}
for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    tr = season <= upto
    va = season == vs
    yv = y[va]
    w = 0.5 ** ((upto - season[tr]) / 2.0)
    n_es = int(tr.sum() * 0.92)
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B)
    print(f'\n=== fold {tag} (train<={upto} -> {vs})  기존8헤드={sc(blend):.1f} ===')
    for nm, auxes in CONFIGS.items():
        Ymat = np.column_stack([y.astype(np.float64)] + auxes)
        ts = time.time()
        m = CatBoostRegressor(**CAT)
        m.fit(X.loc[tr].iloc[:n_es], Ymat[tr][:n_es], sample_weight=w[:n_es],
              eval_set=(X.loc[tr].iloc[n_es:], Ymat[tr][n_es:]))
        p = np.clip(m.predict(X.loc[va])[:, 0], 0, 1)
        np.save(f'dev/cache_strk_{nm}_{tag}.npy', p)
        d = p - blend; dc = d - d.mean()
        V = float(np.mean(dc ** 2)); A = float(np.mean(dc * (blend - yv)))
        rho = -A / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
        results.setdefault(nm, {})[tag] = rho
        print(f'  {nm:<16} BSS={sc(p):8.1f}  rho={rho:+.5f} ({abs(rho)/NEED_RHO*100:5.1f}%)  '
              f's*={-A/V if V>1e-14 else 0:+.4f}  최대이득={K*A**2/V if V>1e-14 else 0:+6.2f} '
              f'({time.time()-ts:.0f}s)')

print(f'\n=== 종합 (부호 일치 + fold A 크기 순) ===')
print(f'{"config":<16}{"foldA rho":>12}{"foldC rho":>12}{"부호일치":>10}')
for nm in CONFIGS:
    a, c = results[nm]['A'], results[nm]['C']
    ok = 'O' if np.sign(a) == np.sign(c) else 'X'
    print(f'{nm:<16}{a:>+12.5f}{c:>+12.5f}{ok:>10}')
best = max((nm for nm in CONFIGS if np.sign(results[nm]['A']) == np.sign(results[nm]['C'])),
           key=lambda nm: abs(results[nm]['A']), default=None)
print(f'\n추천(부호일치 중 foldA 최대): {best}')
log('완료')
