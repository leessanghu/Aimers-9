"""갈래 A: mc6와 CatBoost멀티태스크(multires/midother/...) 메커니즘을 결합/변형한 3종.
서로 다른 모델객체/캐시파일로 완전히 독립 실행 -> 순서 상관없이 재개 가능(체크포인트).

A1 mc6aux   : MultiRMSE([y, onehot(c0..c5)]) - head0=y 직접회귀, 6클래스는 보조타겟.
              추론 head0만 사용 -> Rule4 안전. 목적함수가 Brier(제곱오차)와 정확히 일치.
A2 mc6brier : MultiRMSE(onehot(c0..c5)) - 6클래스 자체를 제곱오차로 직접학습, head3+4+5 합산.
A3 mc4      : MultiClass(4) - 실패측(middle/reverse/wild)을 1개 클래스로 통합한 ablation.
              성공측 3클래스만 유지. mc6(6클래스) 대비 실패측 세분화가 실제 기여하는지 검증.

전부 mc6와 동일한 라벨정의(build_v112_mc6.py 기준) 사용. fold A(2023->2024)/C(2021->2022).
v117 전체블렌드(mc6=0.48+strk=0.10+8헤드=0.42) 기준 rho + fold A H1/H2 클린맥스게인 + mc6와의 d상관.
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier, CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
NEED_RHO = 0.01740
CD = 'dev/mc6family_cache'
os.makedirs(CD, exist_ok=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
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

cls = np.full(n, -1, dtype=np.int64)          # mc6 라벨(build_v112_mc6.py와 동일)
cls[valid & (mid > 0.5)] = 0                   # middle
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1     # reverse
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y == 0)] = 2                         # wild
cls[nd & (y == 1) & (ball > 0.5)] = 3          # succ_ball
cls[nd & (y == 1) & (strike > 0.5)] = 4        # succ_strk
cls[nd & (y == 1) & (inplay > 0.5)] = 5        # succ_play
SUCC6 = [3, 4, 5]
cls4 = np.where(cls < 0, -1, np.where(cls <= 2, 0, cls - 2))   # 0=fail통합,1/2/3=succ
SUCC4 = [1, 2, 3]
log('클래스 분포(mc6): ' + '  '.join(f'{c}:{(cls==c).mean()*100:.1f}%' for c in range(6)))

HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
S_MC6, S_STRK = 0.48, 0.10


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


def diagnose(name, tag, p, blend_v117, yv, mc6_p, strk_p):
    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
    resid = yv - blend_v117
    d = p - blend_v117; dc = d - d.mean()
    V = float(np.mean(dc ** 2)); A = float(np.mean(dc * (blend_v117 - yv)))
    rho = -A / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
    gain = K * A ** 2 / V if V > 1e-14 else 0.0
    d_mc6 = mc6_p - blend_v117; d_mc6 -= d_mc6.mean()
    corr_mc6 = float(np.mean(dc * d_mc6) / np.sqrt(V * np.mean(d_mc6**2) + 1e-18))
    print(f'  [{name}/{tag}] 단독BSS={sc(p):8.2f}  v117기준rho={rho:+.5f}({abs(rho)/NEED_RHO*100:5.1f}%)  '
          f'로컬최대이득={gain:+7.2f}  mc6와d상관={corr_mc6:+.3f}')
    return dict(rho=rho, gain=gain, corr_mc6=corr_mc6, d=d)


CB_REG = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
              loss_function='MultiRMSEWithMissingValues', early_stopping_rounds=50, random_seed=42)
CB_CLS4 = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
               loss_function='MultiClass', classes_count=4, early_stopping_rounds=50, random_seed=42)

results = {}
# fold C는 전역편향으로 정보없음이 확정(2026-08-30) -> fold A만 학습
for tag, upto, vs in [('A', 2023, 2024)]:
    tr = (season <= upto)
    va = season == vs
    yv = y[va]
    w = 0.5 ** ((upto - season[tr]) / 2.0)

    H = build8(tag)
    W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
    t_ = sum(W0.values()); W0 = {k: v / t_ for k, v in W0.items()}
    blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
    mc6_p = np.load(f'dev/cache_mc6head_{tag}.npy')
    strk_p = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    blend_v117 = np.clip(rest * blend8 + S_MC6 * mc6_p + S_STRK * strk_p, 0, 1)

    print(f'\n{"="*90}\n=== fold {tag} (train<={upto} -> {vs}) ===\n{"="*90}')

    # --- A1: mc6aux, MultiRMSE([y, onehot6]) ---
    f1 = f'{CD}/{tag}_mc6aux.npy'
    if os.path.exists(f1):
        p1 = np.load(f1)
    else:
        onehot_tr = np.full((tr.sum(), 6), np.nan)
        ctr = cls[tr]
        okc = ctr >= 0
        for c in range(6):
            onehot_tr[okc, c] = (ctr[okc] == c).astype(np.float64)
        Ymat = np.column_stack([y[tr], onehot_tr])
        n_es = int(tr.sum() * 0.92)
        ts = time.time()
        m1 = CatBoostRegressor(**CB_REG)
        m1.fit(X.loc[tr].iloc[:n_es], Ymat[:n_es], sample_weight=w[:n_es],
               eval_set=(X.loc[tr].iloc[n_es:], Ymat[n_es:]))
        p1 = np.clip(m1.predict(X.loc[va])[:, 0], 0, 1)
        np.save(f1, p1)
        log(f'[{tag}] A1(mc6aux) 학습완료 best_iter={m1.best_iteration_} ({time.time()-ts:.0f}s)')
    r1 = diagnose('A1_mc6aux', tag, p1, blend_v117, yv, mc6_p, strk_p)

    # --- A2: mc6brier, MultiRMSE(onehot6만) ---
    f2 = f'{CD}/{tag}_mc6brier.npy'
    if os.path.exists(f2):
        p2 = np.load(f2)
    else:
        onehot_tr2 = np.full((tr.sum(), 6), np.nan)
        ctr = cls[tr]
        okc = ctr >= 0
        for c in range(6):
            onehot_tr2[okc, c] = (ctr[okc] == c).astype(np.float64)
        n_es = int(tr.sum() * 0.92)
        ts = time.time()
        m2 = CatBoostRegressor(**CB_REG)
        m2.fit(X.loc[tr].iloc[:n_es], onehot_tr2[:n_es], sample_weight=w[:n_es],
               eval_set=(X.loc[tr].iloc[n_es:], onehot_tr2[n_es:]))
        proba2 = np.clip(m2.predict(X.loc[va]), 0, 1)
        p2 = np.clip(proba2[:, SUCC6].sum(axis=1), 0, 1)
        np.save(f2, p2)
        log(f'[{tag}] A2(mc6brier) 학습완료 best_iter={m2.best_iteration_} ({time.time()-ts:.0f}s)')
    r2 = diagnose('A2_mc6brier', tag, p2, blend_v117, yv, mc6_p, strk_p)

    # --- A3: mc4, MultiClass(4) - 실패측 통합 ablation ---
    f3 = f'{CD}/{tag}_mc4.npy'
    if os.path.exists(f3):
        p3 = np.load(f3)
    else:
        m3tr = cls4[tr] >= 0
        Xtr3, ctr3, wtr3 = X.loc[tr][m3tr], cls4[tr][m3tr], w[m3tr]
        n_es = int(len(Xtr3) * 0.92)
        ts = time.time()
        m3 = CatBoostClassifier(**CB_CLS4)
        m3.fit(Xtr3.iloc[:n_es], ctr3[:n_es], sample_weight=wtr3[:n_es],
               eval_set=(Xtr3.iloc[n_es:], ctr3[n_es:]))
        proba3 = m3.predict_proba(X.loc[va])
        p3 = np.clip(proba3[:, SUCC4].sum(axis=1), 0, 1)
        np.save(f3, p3)
        log(f'[{tag}] A3(mc4) 학습완료 best_iter={m3.best_iteration_} ({time.time()-ts:.0f}s)')
    r3 = diagnose('A3_mc4', tag, p3, blend_v117, yv, mc6_p, strk_p)

    results[tag] = dict(A1=r1, A2=r2, A3=r3)

    # fold A에서만: H1/H2 클린 max-gain(대조군 포함)
    if tag == 'A':
        Xv = X.loc[va]
        mth = Xv['game_month'].to_numpy()
        H1m = mth <= 6; H2m = ~H1m
        resid = yv - blend_v117
        sc2 = lambda pp, msk: 1e5 * (1 - np.mean((np.clip(pp[msk], 0, 1) - yv[msk]) ** 2) / B_)
        rng = np.random.RandomState(8)

        def honest(dd):
            gg = []
            for fit_m, ev_m in [(H1m, H2m), (H2m, H1m)]:
                mdf = dd[fit_m].mean()
                cvv = np.mean((dd[fit_m]-mdf)*(resid[fit_m]-resid[fit_m].mean()))
                vr = np.mean((dd[fit_m]-mdf)**2)
                a = cvv/vr if vr > 1e-14 else 0.0
                bl = blend_v117.copy()
                bl[ev_m] = blend_v117[ev_m] + a*(dd[ev_m]-mdf)
                gg.append(sc2(bl, ev_m) - sc2(blend_v117, ev_m))
            return gg

        print(f'\n  --- fold A 클린 H1/H2 max-gain (v117 기준) ---')
        ctrl = rng.normal(0, r1['d'].std(), len(yv))
        gc = honest(ctrl)
        print(f'    대조군      H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
        for nm, r in [('A1_mc6aux', r1), ('A2_mc6brier', r2), ('A3_mc4', r3)]:
            g = honest(r['d'])
            print(f'    {nm:<12} H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')

print(f'\n{"="*90}\n=== 종합 (fold A 기준) ===\n{"="*90}')
for nm in ('A1', 'A2', 'A3'):
    for tg in results:
        if nm in results[tg]:
            r = results[tg][nm]
            print(f'  {nm}/{tg}: rho={r["rho"]:+.5f}  gain={r["gain"]:+.2f}  corr_mc6={r["corr_mc6"]:+.3f}')
log('전체 완료')
