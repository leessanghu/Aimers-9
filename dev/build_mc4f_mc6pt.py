"""갈래 A 후속: mc4f(2x2 ablation 완성) + mc6pt(성공측 구종축 분할).

mc4f  : MultiClass(4) = middle/reverse/wild/succ(통합) - 실패측만 분할, 성공측 통합.
        mc4(A3, 성공측만분할)의 여집합. mc6=성공+실패 둘다분할과 함께 2x2 완성.
        가설: 성공측 분할이 핵심이면 mc4≈mc6 >> mc4f, mc4f≈base(binary) 나올 것.
mc6pt : MultiClass(6) = middle/reverse/wild/succ_fast/succ_break/succ_off
        mc6(판정축 ball/strk/play)의 자매모델. 구종축(fastball/breaking/offspeed)으로
        성공측을 재분할. 구종축은 mc5 헤드 하나만 의미있게 쓰고 있어서 여지가 있다는
        importance 감사결과에 근거.

전부 mc6와 동일 mid/reverse 라벨정의, ptype은 dev/recovered_pitch_type.npy(코드체계
0=fastball/1=breaking/2=offspeed, -1=무효) 사용.
"""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
NEED_RHO = 0.01740
CD = 'dev/mc4f_mc6pt_cache'
os.makedirs(CD, exist_ok=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')
ptype = np.load('dev/recovered_pitch_type.npy')
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
middle = valid & (mid > 0.5)
reverse = valid & (rev > 0.5) & (mid < 0.5)
nd = valid & (mid < 0.5) & (rev < 0.5)   # not-dangerous (wild/succ 후보)

# --- mc4f: middle/reverse/wild/succ(통합) ---
cls4f = np.full(n, -1, dtype=np.int64)
cls4f[middle] = 0
cls4f[reverse] = 1
cls4f[nd & (y == 0)] = 2
cls4f[nd & (y == 1)] = 3
SUCC4f = [3]
log('클래스 분포(mc4f): ' + '  '.join(f'{c}:{(cls4f==c).mean()*100:.1f}%' for c in range(4))
    + f'  미분류:{(cls4f<0).mean()*100:.2f}%')

# --- mc6pt: middle/reverse/wild/succ_fast/succ_break/succ_off ---
ok_pt = ptype >= 0
is_fb, is_bk, is_os = ptype == 0, ptype == 1, ptype == 2
cls6pt = np.full(n, -1, dtype=np.int64)
cls6pt[middle] = 0
cls6pt[reverse] = 1
cls6pt[nd & (y == 0)] = 2
succ_ok = nd & (y == 1) & ok_pt
cls6pt[succ_ok & is_fb] = 3
cls6pt[succ_ok & is_bk] = 4
cls6pt[succ_ok & is_os] = 5
SUCC6pt = [3, 4, 5]
log('클래스 분포(mc6pt): ' + '  '.join(f'{c}:{(cls6pt==c).mean()*100:.1f}%' for c in range(6))
    + f'  미분류:{(cls6pt<0).mean()*100:.2f}%')

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


def diagnose(name, tag, p, blend_v117, yv, mc6_p):
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


CB4 = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='MultiClass', classes_count=4, early_stopping_rounds=50, random_seed=42)
CB6 = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='MultiClass', classes_count=6, early_stopping_rounds=50, random_seed=42)

results = {}
# fold C는 전역편향으로 정보없음이 확정(2026-08-30) -> fold A만 학습
for tag, upto, vs in [('A', 2023, 2024)]:
    tr = season <= upto
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

    # --- mc4f ---
    f4 = f'{CD}/{tag}_mc4f.npy'
    if os.path.exists(f4):
        p4 = np.load(f4)
    else:
        m4tr = cls4f[tr] >= 0
        Xtr4, ctr4, wtr4 = X.loc[tr][m4tr], cls4f[tr][m4tr], w[m4tr]
        n_es = int(len(Xtr4) * 0.92)
        ts = time.time()
        m4 = CatBoostClassifier(**CB4)
        m4.fit(Xtr4.iloc[:n_es], ctr4[:n_es], sample_weight=wtr4[:n_es],
               eval_set=(Xtr4.iloc[n_es:], ctr4[n_es:]))
        proba4 = m4.predict_proba(X.loc[va])
        p4 = np.clip(proba4[:, SUCC4f].sum(axis=1), 0, 1)
        np.save(f4, p4)
        log(f'[{tag}] mc4f 학습완료 best_iter={m4.best_iteration_} ({time.time()-ts:.0f}s)')
    r4 = diagnose('mc4f', tag, p4, blend_v117, yv, mc6_p)

    # --- mc6pt ---
    f6 = f'{CD}/{tag}_mc6pt.npy'
    if os.path.exists(f6):
        p6 = np.load(f6)
    else:
        m6tr = cls6pt[tr] >= 0
        Xtr6, ctr6, wtr6 = X.loc[tr][m6tr], cls6pt[tr][m6tr], w[m6tr]
        n_es = int(len(Xtr6) * 0.92)
        ts = time.time()
        m6 = CatBoostClassifier(**CB6)
        m6.fit(Xtr6.iloc[:n_es], ctr6[:n_es], sample_weight=wtr6[:n_es],
               eval_set=(Xtr6.iloc[n_es:], ctr6[n_es:]))
        proba6 = m6.predict_proba(X.loc[va])
        p6 = np.clip(proba6[:, SUCC6pt].sum(axis=1), 0, 1)
        np.save(f6, p6)
        log(f'[{tag}] mc6pt 학습완료 best_iter={m6.best_iteration_} ({time.time()-ts:.0f}s)')
    r6 = diagnose('mc6pt', tag, p6, blend_v117, yv, mc6_p)

    results[tag] = dict(mc4f=r4, mc6pt=r6)

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
        ctrl = rng.normal(0, r4['d'].std(), len(yv))
        gc = honest(ctrl)
        print(f'    대조군    H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
        for nm, r in [('mc4f', r4), ('mc6pt', r6)]:
            g = honest(r['d'])
            print(f'    {nm:<10} H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')

print(f'\n{"="*90}\n=== 종합 (fold A 기준) ===\n{"="*90}')
for nm in ('mc4f', 'mc6pt'):
    for tg in results:
        if nm in results[tg]:
            r = results[tg][nm]
            print(f'  {nm}/{tg}: rho={r["rho"]:+.5f}  gain={r["gain"]:+.2f}  corr_mc6={r["corr_mc6"]:+.3f}')
log('전체 완료')
