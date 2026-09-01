"""mc6 v2 — 계층분해(hurdle 방식). 6-way joint softmax(실패) 대신 이미 검증된
hurdle/ordinal 패턴대로 단계별 이진/조건부 CatBoost multi-task로 재구성.

진단 결과(빠른 HGB, fold A/C 재현): is_wild AUC .59/.60, succ_ball AUC .58/.58,
strike-vs-play AUC .63/.62 — 개별로는 전부 신호 있음. 6-way joint에서 정확도가
기준선 근처였던 건 타겟이 예측불가능해서가 아니라 조인트 학습이 신호를 죽인 것.

새 구조 (전부 MultiRMSEWithMissingValues, 결측=해당없음 행):
  headA: [y, is_wild]                     전체 행 (fail vs success 이미 y가 앎,
                                            wild은 nd&fail 에서만 유효)
  headB: [y, is_succball]                  nd&y==1 행에서만 유효 (ball vs strike+play)
  headC: [y, is_strike_amongnonball]       nd&y==1&~ball 행에서만 유효 (strike vs play)
  (fail쪽 middle/reverse는 이미 hurdle/ordinal이 다루고 있어 여기선 재복제 안 함)

3개를 각각 학습해서 head0(y)들의 잔차상관을 개별로 재고, 마지막에 평균앙상블까지 본다.
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
nd = valid & (mid < 0.5) & (rev < 0.5)

is_wild = np.where(nd, (y == 0).astype(np.float64), np.nan)
is_succball = np.where(nd & (y == 1), ball.astype(np.float64), np.nan)
notball = nd & (y == 1) & (ball < 0.5)
is_strike = np.where(notball, strike.astype(np.float64), np.nan)

TARGETS = {
    'headA_wild': is_wild,
    'headB_ball': is_succball,
    'headC_strike': is_strike,
}
for nm, t in TARGETS.items():
    print(f'  {nm}: 유효 {np.isfinite(t).sum():,} ({np.isfinite(t).mean()*100:.1f}%)')

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
    n_es = int(tr.sum() * 0.92)

    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B)

    preds = {}
    for nm, tgt in TARGETS.items():
        Ymat = np.column_stack([y.astype(np.float64), tgt])
        Xtr, Ytr = X.loc[tr], Ymat[tr]
        ts = time.time()
        m = CatBoostRegressor(**CAT)
        m.fit(Xtr.iloc[:n_es], Ytr[:n_es], sample_weight=w[:n_es],
              eval_set=(Xtr.iloc[n_es:], Ytr[n_es:]))
        p = np.clip(m.predict(X.loc[va])[:, 0], 0, 1)
        preds[nm] = p
        log(f'[{tag}/{nm}] 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')

        d = p - blend
        dc = d - d.mean()
        V = float(np.mean(dc ** 2)); C = float(np.mean(dc * resid))
        rho = C / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
        print(f'  {nm:<14} 단독BSS={sc(p):8.1f}  rho={rho:+.5f} '
              f'(필요치의 {abs(rho)/NEED_RHO*100:5.1f}%)  최대이득={K*C**2/V if V>1e-14 else 0:+6.2f}')
        np.save(f'dev/cache_mc6h_{nm}_{tag}.npy', p)

    # 3개 평균
    p_avg = np.mean(list(preds.values()), axis=0)
    d = p_avg - blend; dc = d - d.mean()
    V = float(np.mean(dc ** 2)); C = float(np.mean(dc * resid))
    rho = C / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
    print(f'  {"3개 평균":<14} 단독BSS={sc(p_avg):8.1f}  rho={rho:+.5f} '
        f'(필요치의 {abs(rho)/NEED_RHO*100:5.1f}%)  최대이득={K*C**2/V if V>1e-14 else 0:+6.2f}')
    return preds, blend, resid, yv


log('=== fold A ===')
predsA, blendA, residA, yvA = run(2023, 2024, 'A')
log('=== fold C ===')
run(2021, 2022, 'C')

log('클린검증(H1/H2 + 대조군, headA 기준)...')
Xv = X.loc[season == 2024]
mth = Xv['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
sc2 = lambda pp, msk: 1e5 * (1 - np.mean((np.clip(pp[msk], 0, 1) - yvA[msk]) ** 2) / B)
p_avg = np.mean(list(predsA.values()), axis=0)
d = p_avg - blendA
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
print(f'\n=== 클린 max-gain (fold A, 3헤드평균 vs 기존8헤드 블렌드) ===')
print(f'  대조군          H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
print(f'  mc6_hier(평균)  H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')
log('완료')
