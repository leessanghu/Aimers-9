"""mc6 순수분할 헤드: 성공을 '부정형(실패가 아닌 것)'이 아니라 '양의 정의'로 재현.

공식 실패 3유형 + 성공 3하위유형의 완전분할 (합 99.95%, 모든 클래스가 순수):
  class0 middle    14.96%  성공률 0%    <- 공식실패 1 (가운데 몰림)
  class1 reverse   19.48%  성공률 0%    <- 공식실패 3 (반대방향)
  class2 wild      13.17%  성공률 0%    <- 공식실패 2 (크게 벗어남) *기존 mc5는 성공과 섞어놨음*
  class3 succ_ball 15.67%  성공률 100%  <- 포수가 존밖 요구, 적중
  class4 succ_strk 26.06%  성공률 100%  <- 존안 요구, 적중
  class5 succ_play 10.61%  성공률 100%  <- 타자가 침

=> P(success) = P(3)+P(4)+P(5)  (정확한 항등식. mc5처럼 succ_by_cls 가중 불필요)

기존 mc5와의 차이:
  mc5: class2=nd&ball(성공률 59.1%), class3=nd&strike, class4=nd&inplay
       -> 공식실패 2번이 성공과 같은 클래스에 뭉쳐 있어 분리 안 됨
  mc6: wild을 독립 클래스로 분리 + 성공을 3하위유형으로 분해

Rule4: 모든 라벨은 as-of 카운터 차분으로 학습데이터에서만 복원. 추론시엔 확률합만 사용.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

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
call = np.load('dev/recovered_call_axis.npy')      # [ball, strike, inplay]
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

# ---- 라벨 복원 (mid/rev) ----
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
cls[valid & (mid > 0.5)] = 0                                    # middle
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1                      # reverse
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y == 0)] = 2                                          # wild (공식실패2)
cls[nd & (y == 1) & (ball > 0.5)] = 3                           # 성공-존밖
cls[nd & (y == 1) & (strike > 0.5)] = 4                         # 성공-존안
cls[nd & (y == 1) & (inplay > 0.5)] = 5                         # 성공-인플레이

names = ['middle', 'reverse', 'wild', 'succ_ball', 'succ_strk', 'succ_play']
print('=== mc6 클래스 분포 및 순수성 검증 ===')
for c in range(6):
    m = cls == c
    print(f'  class{c} {names[c]:<11} n={m.sum():>9,} ({m.mean()*100:5.2f}%)  성공률={y[m].mean()*100:6.2f}%')
print(f'  미분류(-1): {(cls<0).sum():,} ({(cls<0).mean()*100:.2f}%)')
SUCC_CLASSES = [3, 4, 5]

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=6, early_stopping_rounds=50,
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
    tr = (season <= upto) & (cls >= 0)
    va = season == vs
    yv = y[va]
    w = 0.5 ** ((upto - season[tr]) / 2.0)
    Xtr, ctr = X.loc[tr], cls[tr]
    n_es = int(len(Xtr) * 0.92)
    ts = time.time()
    m = CatBoostClassifier(**CB)
    m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=w[:n_es],
          eval_set=(Xtr.iloc[n_es:], ctr[n_es:]))
    proba = m.predict_proba(X.loc[va])
    p = np.clip(proba[:, SUCC_CLASSES].sum(axis=1), 0, 1)
    log(f'[{tag}] 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
    np.save(f'dev/cache_mc6head_{tag}.npy', p)

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B)
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  mc6_head 단독 BSS = {sc(p):.2f}   (P = class3+4+5)')
    cv = cls[va]
    okv = cv >= 0
    if okv.sum() > 0:
        acc = (np.argmax(proba, axis=1)[okv] == cv[okv]).mean()
        base = max((cv[okv] == c).mean() for c in range(6))
        print(f'  6-class 정확도 = {acc*100:.2f}%  (최빈값 기준선 {base*100:.2f}%)')

    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    d = p - blend
    dc = d - d.mean()
    V = float(np.mean(dc ** 2)); C = float(np.mean(dc * resid))
    rho = C / np.sqrt(V * float(np.mean(resid ** 2)))
    print(f'  기존8헤드 블렌드 BSS = {sc(blend):.2f}')
    print(f'  잔차상관 rho = {rho:+.5f}   (+30점 필요치 {NEED_RHO:.5f}의 {abs(rho)/NEED_RHO*100:.1f}%)')
    print(f'  최대이득 = {K*C**2/V:+.2f}점   최적가중치 s* = {-C/V:+.4f}')
    return p, blend, resid, yv


log('=== fold A ===')
pA, blendA, residA, yvA = run(2023, 2024, 'A')
log('=== fold C ===')
run(2021, 2022, 'C')

log('클린검증(H1/H2 + 대조군)...')
Xv = X.loc[season == 2024]
mth = Xv['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
sc2 = lambda pp, msk: 1e5 * (1 - np.mean((np.clip(pp[msk], 0, 1) - yvA[msk]) ** 2) / B)
d = pA - blendA
rng = np.random.RandomState(8)
ctrl = rng.normal(0, d.std(), len(yvA))


def honest(dd):
    g = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        mdf = dd[fit_m].mean()
        cvv = np.mean((dd[fit_m]-mdf)*(residA[fit_m]-residA[fit_m].mean()))
        vr = np.mean((dd[fit_m]-mdf)**2)
        a = cvv/vr if vr > 1e-14 else 0.0
        bl = blendA.copy()
        bl[ev_m] = blendA[ev_m] + a*(dd[ev_m]-mdf)
        g.append(sc2(bl, ev_m) - sc2(blendA, ev_m))
    return g


gc = honest(ctrl); g = honest(d)
print(f'\n=== 클린 max-gain (fold A, 기존8헤드 블렌드 대비) ===')
print(f'  대조군     H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
print(f'  mc6_head   H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')
log('완료')
