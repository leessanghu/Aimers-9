"""신규 헤드 실험: '시퀀스/상태' 보조타겟.

mc6/hurdle이 통한 원리:
  - 라벨을 as-of 카운터 차분으로 복원 -> 학습전용 보조타겟
  - 추론시엔 head0(y)만 사용 -> Rule4 안전
  - 모델이 '중간구조'를 명시적으로 학습하게 강제 -> head0 표현 개선

지금까지 쓴 보조타겟은 전부 '이 투구 하나의 결과'(reverse/middle/ball/strike/구종).
아직 안 쓴 축 = '직전 투구와의 관계' (시퀀스 상태).

보조타겟 3종 (전부 학습데이터에서만 복원, test에선 계산 불가 -> 보조타겟 전용):
  seqA: prev_y      직전 투구 성공 여부
  seqB: fail_streak 직전까지 연속실패 길이 (clip 0~3, 정규화)
  seqC: prev_ball   직전 투구가 볼이었나

각각 [y, aux] multi-task로 학습, fold A/C honest 검증 + 대조군.
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

# 직전 투구 복원 (같은 투수 & asof_n 정확히 +1)
same_prev = np.zeros(len(o), dtype=bool)
same_prev[1:] = (pid[o][1:] == pid[o][:-1])
dn_ord = np.full(len(o), np.nan)
dn_ord[1:] = n_[o][1:] - n_[o][:-1]
valid_prev_ord = same_prev & (dn_ord == 1)

prev_y_ord = np.full(len(o), np.nan); prev_y_ord[1:] = y[o][:-1]
prev_y = np.full(n, np.nan); prev_y[o] = np.where(valid_prev_ord, prev_y_ord, np.nan)

ball = call[:, 0]
prev_ball_ord = np.full(len(o), np.nan); prev_ball_ord[1:] = ball[o][:-1]
prev_ball = np.full(n, np.nan); prev_ball[o] = np.where(valid_prev_ord, prev_ball_ord, np.nan)

# 연속실패 길이 (벡터화: 실패면 누적, 성공이면 리셋)
yo = y[o]
streak_ord = np.zeros(len(o))
cur = 0.0
for i in range(len(o)):
    if not valid_prev_ord[i]:
        cur = 0.0
    streak_ord[i] = cur
    cur = 0.0 if yo[i] == 1 else cur + 1
fail_streak = np.empty(n); fail_streak[o] = streak_ord
fail_streak = np.where(np.isfinite(prev_y), np.clip(fail_streak, 0, 3) / 3.0, np.nan)

TARGETS = {
    'seqA_prev_y': prev_y,
    'seqB_streak': fail_streak,
    'seqC_prev_ball': prev_ball,
}
for nm, t in TARGETS.items():
    log(f'{nm}: 유효 {np.isfinite(t).sum():,} ({np.isfinite(t).mean()*100:.1f}%)')

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


def run(upto, vs, tag):
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
    preds = {}
    for nm, tgt in TARGETS.items():
        Ymat = np.column_stack([y.astype(np.float64), tgt])
        ts = time.time()
        m = CatBoostRegressor(**CAT)
        m.fit(X.loc[tr].iloc[:n_es], Ymat[tr][:n_es], sample_weight=w[:n_es],
              eval_set=(X.loc[tr].iloc[n_es:], Ymat[tr][n_es:]))
        p = np.clip(m.predict(X.loc[va])[:, 0], 0, 1)
        preds[nm] = p
        np.save(f'dev/cache_seq_{nm}_{tag}.npy', p)
        d = p - blend; dc = d - d.mean()
        V = float(np.mean(dc ** 2)); A = float(np.mean(dc * (blend - yv)))
        rho = -A / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
        print(f'  {nm:<16} BSS={sc(p):8.1f}  rho={rho:+.5f} ({abs(rho)/NEED_RHO*100:5.1f}%)  '
              f's*={-A/V if V>1e-14 else 0:+.4f}  최대이득={K*A**2/V if V>1e-14 else 0:+6.2f} '
              f'({time.time()-ts:.0f}s)')
    p_avg = np.mean(list(preds.values()), axis=0)
    d = p_avg - blend; dc = d - d.mean()
    V = float(np.mean(dc ** 2)); A = float(np.mean(dc * (blend - yv)))
    rho = -A / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
    print(f'  {"3개평균":<16} BSS={sc(p_avg):8.1f}  rho={rho:+.5f} ({abs(rho)/NEED_RHO*100:5.1f}%)  '
          f's*={-A/V if V>1e-14 else 0:+.4f}  최대이득={K*A**2/V if V>1e-14 else 0:+6.2f}')
    return preds, blend, resid, yv


log('=== fold A ===')
pA, blendA, residA, yvA = run(2023, 2024, 'A')
log('=== fold C ===')
run(2021, 2022, 'C')

log('클린검증(대조군 포함)...')
Xv = X.loc[season == 2024]
mth = Xv['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
sc2 = lambda pp, msk: 1e5 * (1 - np.mean((np.clip(pp[msk], 0, 1) - yvA[msk]) ** 2) / B)
p_avg = np.mean(list(pA.values()), axis=0)
d = p_avg - blendA
rng = np.random.RandomState(8)
ctrl = rng.normal(0, d.std(), len(yvA))


def honest(dd):
    gg = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        mdf = dd[fit_m].mean()
        cvv = np.mean((dd[fit_m]-mdf)*(residA[fit_m]-residA[fit_m].mean()))
        vr = np.mean((dd[fit_m]-mdf)**2)
        a = cvv/vr if vr > 1e-14 else 0.0
        bl = blendA.copy()
        bl[ev_m] = blendA[ev_m] + a*(dd[ev_m]-mdf)
        gg.append(sc2(bl, ev_m) - sc2(blendA, ev_m))
    return gg


gc = honest(ctrl); gv = honest(d)
print(f'\n=== 클린 max-gain (fold A, 3헤드평균) ===')
print(f'  대조군    H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
print(f'  seq_head  H1->H2={gv[0]:+7.2f}  H2->H1={gv[1]:+7.2f}  평균={np.mean(gv):+7.2f}')
log('완료')
