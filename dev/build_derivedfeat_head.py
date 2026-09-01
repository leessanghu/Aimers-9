"""f4(succball|count,pt), f5(wild|pt) 파생피처를 입력으로 추가한 헤드.
피처 자체를 블렌드에 바로 더하는 게 아니라(그건 위 스크린 단계일뿐),
162피처 + f4,f5를 입력으로 받는 CatBoost가 [y, wild, succ_ball] multi-task로 학습.
=> 피처(레버1)와 타겟분해(레버2)를 동시에 쓰는 구조.

Rule4: f4/f5는 train<=upto 데이터로 만든 (count,구종)/(구종) 조건부 테이블을
test에서 자기 행의 balls/strikes/구종추정으로 조회 -> 안전.
단, '구종추정'은 test에서 알 수 없으므로(투구단위 라벨) f5는 '구종믹스 기대값'으로 근사:
  f5_expected(row) = sum_t P(t|count) * [wild_rate(t) - global_wild]
이건 이미 probe_pitchtype_derived_feature.py에서 시도해 fold간 부호불일치로 실패했던
'구종별 성공률 기댓값'과 다르다 - 여기는 '구종별 wild율'이고 투수별로 안 쪼갠다.

fold A/C honest 검증 + 대조군.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

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
call = np.load('dev/recovered_call_axis.npy')
ptype = np.load('dev/recovered_pitch_type.npy')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
cs = (df['balls_before'].to_numpy() * 4 + df['strikes_before'].to_numpy())
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
wild = (nd & (y == 0)).astype(np.float64)
succ_ball_lab = np.where(nd, (y == 1).astype(np.float64) * ball, np.nan)
wild_lab = np.where(valid, wild, np.nan)

ok_pt = ptype >= 0


def build_pt_features(upto, va_mask):
    """f_wild_pt(row) = wild_rate(구종) - global_wild   (train<=upto)
       f_succball_cntpt(row) = P(succ_ball|count,구종) - global   (train<=upto)"""
    trv = (season <= upto) & valid
    g_wild = float(wild[trv].mean())
    wtab = pd.DataFrame({'t': ptype[trv], 'w': wild[trv]})
    wrate = wtab.groupby('t')['w'].mean()
    f5 = pd.Series(ptype[va_mask]).map(wrate).fillna(g_wild).to_numpy(np.float64) - g_wild

    trn = (season <= upto) & nd
    sbtab = pd.DataFrame({'cs': cs[trn], 't': ptype[trn], 'y': y[trn], 'ball': ball[trn]})
    sbtab['succ_ball'] = (sbtab['y'] == 1).astype(float) * sbtab['ball']
    grp = sbtab.groupby(['cs', 't'])['succ_ball'].mean()
    g_sb = float(sbtab['succ_ball'].mean())
    key = pd.MultiIndex.from_arrays([cs[va_mask], ptype[va_mask]])
    f4 = grp.reindex(key).fillna(g_sb).to_numpy(np.float64) - g_sb
    return f4, f5


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

    f4_all = np.zeros(n); f5_all = np.zeros(n)
    f4_tr, f5_tr = build_pt_features(upto, tr)   # train쪽은 walk-forward 아님(단순근사, 스크리닝용)
    f4_va, f5_va = build_pt_features(upto, va)
    f4_all[tr] = f4_tr; f5_all[tr] = f5_tr
    f4_all[va] = f4_va; f5_all[va] = f5_va

    Xtr = X.loc[tr, FEAT].copy()
    Xtr['derived_f4'] = f4_all[tr]
    Xtr['derived_f5'] = f5_all[tr]
    Xva = X.loc[va, FEAT].copy()
    Xva['derived_f4'] = f4_all[va]
    Xva['derived_f5'] = f5_all[va]

    Ymat = np.column_stack([y.astype(np.float64), wild_lab, succ_ball_lab])
    n_es = int(tr.sum() * 0.92)
    ts = time.time()
    m = CatBoostRegressor(**CAT)
    m.fit(Xtr.iloc[:n_es], Ymat[tr][:n_es], sample_weight=w[:n_es],
          eval_set=(Xtr.iloc[n_es:], Ymat[tr][n_es:]))
    p = np.clip(m.predict(Xva)[:, 0], 0, 1)
    log(f'[{tag}] 학습완료 best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
    np.save(f'dev/cache_derivedhead_{tag}.npy', p)

    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
    d = p - blend; dc = d - d.mean()
    V = float(np.mean(dc ** 2)); A = float(np.mean(dc * (blend - yv)))
    rho = -A / np.sqrt(V * float(np.mean(resid ** 2))) if V > 1e-14 else 0.0
    d_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy') - blend; d_mc6 -= d_mc6.mean()
    corr_mc6 = float(np.mean(dc * d_mc6) / np.sqrt(V * np.mean(d_mc6 ** 2)))
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  derived_head 단독 BSS={sc(p):.1f}  기존8헤드={sc(blend):.1f}')
    print(f'  rho={rho:+.5f} (필요치의 {abs(rho)/NEED_RHO*100:.1f}%)  최대이득(로컬,참고)={K*A**2/V if V>1e-14 else 0:+.2f}')
    print(f'  mc6와 d상관={corr_mc6:+.3f}')
    return p, blend, resid, yv


log('=== fold A ===')
pA, blendA, residA, yvA = run(2023, 2024, 'A')
log('=== fold C ===')
run(2021, 2022, 'C')

log('클린검증(대조군)...')
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
print(f'  대조군       H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
print(f'  derivedhead  H1->H2={gv[0]:+7.2f}  H2->H1={gv[1]:+7.2f}  평균={np.mean(gv):+7.2f}')
log('완료')
