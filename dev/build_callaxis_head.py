"""판정축 multi-task 헤드: Y = [y, ball, strike, inplay]

근거:
 - ball+strike+inplay = 1 이 100.00% 성립하는 깨끗한 3분할 (36.96/44.33/18.71%)
 - 결측 0.1%뿐 (존의도와 달리 MNAR 아님 - 모든 투구에서 관측됨)
 - success=1 & ball=1 이 15.68% 존재 = 포수가 존 밖을 요구한 경우.
   판정축은 이 '포수 의도' 구조를 선택편향 없이 담는 대리변수.
 - 카운트 수준에서 corr(존안요구율, 성공률) = -0.512 (인과사슬 존재)

기존 condball 헤드는 '위험하지 않은 투구에 한해 1-ball'이라 조건부/부분적.
이건 판정축 전체를 무조건부로 학습시킨다.

Rule4: 판정축 라벨은 학습 전용, 추론은 head0만. (hurdle/midother/future50과 동일)
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
call = np.load('dev/recovered_call_axis.npy')   # [ball, strike, inplay], NaN=복원불가
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

Ymat = np.column_stack([y.astype(np.float64), call[:, 0], call[:, 1], call[:, 2]])
okc = np.isfinite(call[:, 0])
log(f'타겟 구성: [y, ball, strike, inplay]  판정축 유효 {okc.mean()*100:.1f}%')

CAT = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
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
    Xtr, Ytr = X.loc[tr], Ymat[tr]
    n_es = int(len(Xtr) * 0.92)
    ts = time.time()
    m = CatBoostRegressor(**CAT)
    m.fit(Xtr.iloc[:n_es], Ytr[:n_es], sample_weight=w[:n_es],
          eval_set=(Xtr.iloc[n_es:], Ytr[n_es:]))
    heads = m.predict(X.loc[va])
    p = np.clip(heads[:, 0], 0, 1)
    log(f'[{tag}] 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
    np.save(f'dev/cache_callaxishead_{tag}.npy', p)

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B)
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  callaxis_head 단독 BSS = {sc(p):.2f}')
    okv = okc[va]
    if okv.sum() > 0:
        pred_c = np.argmax(heads[:, 1:4], axis=1)[okv]
        true_c = np.argmax(call[va][okv], axis=1)
        base_acc = max(np.mean(true_c == i) for i in range(3))
        print(f'  보조헤드 판정 정확도 = {(pred_c == true_c).mean()*100:.2f}% '
              f'(최빈값 기준선 {base_acc*100:.2f}%)')

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
    print(f'  최대이득 = {K*C**2/V:+.2f}점')
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
        cv = np.mean((dd[fit_m]-mdf)*(residA[fit_m]-residA[fit_m].mean()))
        vr = np.mean((dd[fit_m]-mdf)**2)
        a = cv/vr if vr > 1e-14 else 0.0
        bl = blendA.copy()
        bl[ev_m] = blendA[ev_m] + a*(dd[ev_m]-mdf)
        g.append(sc2(bl, ev_m) - sc2(blendA, ev_m))
    return g


gc = honest(ctrl); g = honest(d)
print(f'\n=== 클린 max-gain (fold A, 기존8헤드 블렌드 대비) ===')
print(f'  대조군         H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
print(f'  callaxis_head  H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')
log('완료')
