"""mc10 = mc6를 middle/reverse도 판정축(ball/strike/inplay)으로 재분할한 10분할.

class0 middle&ball    2.73%  성공률0%
class1 middle&strike  7.97%  성공률0%
class2 middle&inplay  4.25%  성공률0%
class3 reverse&ball   7.69%  성공률0%
class4 reverse&strike 8.42%  성공률0%
class5 reverse&inplay 3.37%  성공률0%
class6 wild           13.17% 성공률0%   (미분할 유지 - mc8에서 분할시 mc6와 상관0.72+로 실패)
class7 succ_ball       15.67% 성공률100%
class8 succ_strike     26.06% 성공률100%
class9 succ_inplay     10.61% 성공률100%
P(success) = P(7)+P(8)+P(9)

사전진단(HGB quick AUC, fold A/C): mid_ball 0.590/0.595, mid_strike 0.573/0.579,
rev_ball 0.577/0.577, rev_strike 0.544/0.540 - 전부 재현, mc6/strk 수준.

fold A/C honest 학습 + mc6와의 d상관(독립성) + 클린 max-gain(대조군).
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
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
middle = valid & (mid > 0.5)
reverse = valid & (rev > 0.5) & (mid < 0.5)
nd = valid & (mid < 0.5) & (rev < 0.5)

cls = np.full(n, -1, np.int64)
cls[middle & (ball > 0.5)] = 0
cls[middle & (strike > 0.5)] = 1
cls[middle & (inplay > 0.5)] = 2
cls[reverse & (ball > 0.5)] = 3
cls[reverse & (strike > 0.5)] = 4
cls[reverse & (inplay > 0.5)] = 5
cls[nd & (y == 0)] = 6
cls[nd & (y == 1) & (ball > 0.5)] = 7
cls[nd & (y == 1) & (strike > 0.5)] = 8
cls[nd & (y == 1) & (inplay > 0.5)] = 9
SUCC = [7, 8, 9]
names = ['mid_ball', 'mid_strk', 'mid_play', 'rev_ball', 'rev_strk', 'rev_play',
         'wild', 'succ_ball', 'succ_strk', 'succ_play']
print('=== mc10 클래스 분포 및 순수성 ===')
for c in range(10):
    m = cls == c
    print(f'  {c} {names[c]:<10} n={m.sum():>9,} ({m.mean()*100:5.2f}%)  성공률={y[m].mean()*100:6.2f}%')
print(f'  미분류: {(cls<0).sum():,} ({(cls<0).mean()*100:.2f}%)')

CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=10, early_stopping_rounds=50,
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
    p = np.clip(proba[:, SUCC].sum(axis=1), 0, 1)
    log(f'[{tag}] 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
    np.save(f'dev/cache_mc10head_{tag}.npy', p)

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  mc10_head 단독 BSS = {sc(p):.2f}   (P = class7+8+9)')
    cv = cls[va]; okv = cv >= 0
    if okv.sum() > 0:
        acc = (np.argmax(proba, axis=1)[okv] == cv[okv]).mean()
        base = max((cv[okv] == c).mean() for c in range(10))
        print(f'  10-class 정확도 = {acc*100:.2f}%  (최빈값 기준선 {base*100:.2f}%)')

    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    d = p - blend; dc = d - d.mean()
    V = float(np.mean(dc ** 2)); A = float(np.mean(dc * (blend - yv)))
    rho = -A / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
    d_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy') - blend; d_mc6 -= d_mc6.mean()
    corr_mc6 = float(np.mean(dc * d_mc6) / np.sqrt(V * np.mean(d_mc6 ** 2) + 1e-18))
    print(f'  기존8헤드 블렌드 BSS = {sc(blend):.2f}')
    print(f'  잔차상관 rho = {rho:+.5f}   (+30점 필요치 {NEED_RHO:.5f}의 {abs(rho)/NEED_RHO*100:.1f}%)')
    print(f'  최대이득(로컬,참고) = {K*A**2/V if V>1e-14 else 0:+.2f}점')
    print(f'  mc6와 d상관 = {corr_mc6:+.4f}')
    return p, blend, resid, yv, corr_mc6


log('=== fold A ===')
pA, blendA, residA, yvA, corrA = run(2023, 2024, 'A')
log('=== fold C ===')
pC, blendC, residC, yvC, corrC = run(2021, 2022, 'C')

log('클린검증(대조군, fold A)...')
Xv = X.loc[season == 2024]
mth = Xv['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
sc2 = lambda pp, msk: 1e5 * (1 - np.mean((np.clip(pp[msk], 0, 1) - yvA[msk]) ** 2) / B_)
d = pA - blendA
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
print(f'\n=== 클린 max-gain (fold A) ===')
print(f'  대조군    H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
print(f'  mc10      H1->H2={gv[0]:+7.2f}  H2->H1={gv[1]:+7.2f}  평균={np.mean(gv):+7.2f}')

print(f'\n{"="*70}')
print(f'[최종판정] mc6와의 상관: fold A={corrA:+.4f}  fold C={corrC:+.4f}')
if abs(corrA) < 0.90 and abs(corrC) < 0.90:
    print('  -> 0.90 미만, 독립성 기준 통과. 프로덕션 학습 진행 가능.')
else:
    print('  -> 0.90 이상, mc6와 과도하게 중복. 스킵 권장.')
log('완료')
