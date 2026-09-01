"""구종 multi-task 헤드: Y = [y, is_fastball, is_breaking, is_offspeed]

근거:
 - 구종은 as-of 카운터 차분으로 100% 복원됨(커버리지 99.9%)
 - 구종별 제구성공률 격차 5.48%p (직구 .5451 / 변화구 .4903 / 오프스피드 .5135)
 - 구종은 카운트로 강하게 예측됨 (0-0: 직구58% / 3-0: 직구94%)
 => 구종은 '카운트 -> 구종 -> 제구성공'의 매개변수. 모델이 이 중간구조를 명시적으로
    학습하게 강제하면 head0(y) 표현이 좋아질 수 있다.

Rule4: 구종 라벨은 학습에서만 쓰고 추론시엔 head0만 사용 (기존 hurdle/midother/future50과 동일).
검증: honest fold A/C + v88_final 대비 클린 max-gain(중심화+무절편+대조군) + 잔차상관.
기존 8헤드 블렌드 대비 잔차상관을 재는 게 핵심 (+30점 스펙 = rho 0.0174).
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
ptype = np.load('dev/recovered_pitch_type.npy')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

ok = ptype >= 0
is_fb = np.where(ok, (ptype == 0).astype(np.float64), np.nan)
is_bk = np.where(ok, (ptype == 1).astype(np.float64), np.nan)
is_os = np.where(ok, (ptype == 2).astype(np.float64), np.nan)
Ymat = np.column_stack([y.astype(np.float64), is_fb, is_bk, is_os])
log(f'타겟 구성: [y, fastball, breaking, offspeed]  구종유효 {ok.mean()*100:.1f}%')

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
    np.save(f'dev/cache_pitchtypehead_{tag}.npy', p)

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B)
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  pitchtype_head 단독 BSS = {sc(p):.2f}')
    # 보조헤드가 실제로 구종을 맞추는지 (학습이 의미있었는지 확인)
    okv = ok[va]
    if okv.sum() > 0:
        pred_t = np.argmax(heads[:, 1:4], axis=1)[okv]
        true_t = ptype[va][okv]
        print(f'  보조헤드 구종 정확도 = {(pred_t == true_t).mean()*100:.2f}% '
              f'(최빈값 기준선 {max(np.mean(true_t==0), np.mean(true_t==1), np.mean(true_t==2))*100:.2f}%)')

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
    print(f'  최대이득 = {K*C**2/V:+.2f}점   최적가중치 s* = {-C/V*-1:+.4f}')
    return p, blend, resid, yv


log('=== fold A ===')
pA, blendA, residA, yvA = run(2023, 2024, 'A')
log('=== fold C ===')
pC, blendC, residC, yvC = run(2021, 2022, 'C')

# 클린 max-gain (대조군 포함), fold A
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
print(f'  대조군          H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
print(f'  pitchtype_head  H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')
log('완료')
