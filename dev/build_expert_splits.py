"""전문가 분할 일반화 실험: 각 헤드의 주력 조건피처로 mc6 학습을 쪼개 라우팅 합성.

분할축 3종 (importance 감사 기반):
  same_hand   - mc5의 1위 피처(14.28). 플래툰 조건. ~50:50
  tm_matched  - 측정레짐 분할(F/R과 동형). 트랙맨 없는 행은 tm피처 전부 무효. 68:32
  two_strike  - mc6의 1위 피처 strikes_before(12.06). 투구 의도가 바뀌는 지점. ~70:30

스크리닝: fold A, v126 기준, 직교화 축에 mc6split(이미 통과, 프로브 예정)까지 포함 -
  '리그분할 너머로 새로 더해지는 것'만 측정. 순열대조군 z>2만 후보.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:7.0f}s] {m}', flush=True)

B_ = 0.249807
K = 1e5 / B_
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

X_df = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y_all = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
v95a = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95a['feature_order'])
X = X_df[FEAT].astype(np.float64)
call = np.load('dev/recovered_call_axis.npy')

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


rev = diff_label('asof_pitcher_reverse_rate'); mid = diff_label('asof_pitcher_middle_rate')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
cls = np.full(n, -1, dtype=np.int64)
cls[valid & (mid > 0.5)] = 0
cls[valid & (rev > 0.5) & (mid < 0.5)] = 1
nd = valid & (mid < 0.5) & (rev < 0.5)
cls[nd & (y_all == 0)] = 2
cls[nd & (y_all == 1) & (ball > 0.5)] = 3
cls[nd & (y_all == 1) & (strike > 0.5)] = 4
cls[nd & (y_all == 1) & (inplay > 0.5)] = 5
SUCC = [3, 4, 5]
CB = dict(iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
          loss_function='MultiClass', classes_count=6, early_stopping_rounds=50,
          random_seed=42)

# 분할축 정의 (전체 행에 대한 버킷 배정, 추론시 재현가능한 피처만)
sh = X_df['same_hand'].to_numpy(np.float64)
tmm = X_df['tm_matched'].to_numpy(np.float64)
stk_b = X_df['strikes_before'].to_numpy(np.float64)
SPLITS = {
    'same_hand': (sh > 0.5).astype(np.int64),
    'tm_matched': (tmm > 0.5).astype(np.int64),
    'two_strike': (stk_b >= 2).astype(np.int64),
}

tr_era = season <= 2023
va = season == 2024
yv = y_all[va]

# 스크리닝 준비
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50']
H = dict(
    base=avg([f'dev/phase90_cache/A_base_{q}.npy' for q in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/A_core_{q}.npy')) *
                    np.load(f'dev/phase90_cache/A_snc_{q}.npy') for q in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/A_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/A_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/A_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/A_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/A_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/A_future50_multi_s{s}.npy' for s in (42, 7)]),
)
W0 = {k: float(v95a[f'{k}_weight']) for k in HEADS8}
t_ = sum(W0.values()); W0 = {k: v / t_ for k, v in W0.items()}
core = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
COMPS = dict(core=core,
             mc6=np.load('dev/cache_mc6head_A.npy'),
             strk=np.load('dev/cache_strk_strk_linear_A.npy'),
             xu=np.load('dev/cache_xgbunused_A.npy'),
             xr=np.load('dev/cache_xgbrawid_A.npy'),
             lty=np.load('dev/cache_lt_y_A.npy'))
W126 = dict(core=0.3491, mc6=0.4381, strk=0.1740, xu=-0.0316, xr=0.0354, lty=0.0350)
blend = np.clip(sum(W126[k] * COMPS[k] for k in COMPS), 0, 1)
E_r2 = float(np.mean((yv - blend) ** 2))
sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B_)
BASES = [COMPS[k] - blend for k in ('mc6', 'strk', 'xu', 'xr', 'lty')]
BASES.append(np.load('dev/cache_mc6split_A.npy') - blend)   # 리그분할 축에도 직교화


def orth(dd, bases):
    dp = dd - dd.mean()
    for b in bases:
        b = b - b.mean()
        Vb = float(np.mean(b ** 2))
        if Vb > 1e-16:
            dp = dp - (float(np.mean(dp * b)) / Vb) * b
    return dp - dp.mean()


def screen(name, p):
    d = p - blend; d0 = d - d.mean()
    dp = orth(d, BASES)
    Vp = float(np.mean(dp ** 2)); Ap = float(np.mean(dp * (blend - yv)))
    rho_p = -Ap / np.sqrt(Vp * E_r2) if Vp > 1e-18 else 0.0
    ctrl = []
    for sd_ in range(20):
        rng = np.random.RandomState(19000 + sd_)
        dc = orth(rng.permutation(d0), BASES)
        Vc = float(np.mean(dc ** 2))
        if Vc > 1e-18:
            ctrl.append(-float(np.mean(dc * (blend - yv))) / np.sqrt(Vc * E_r2))
    ctrl = np.array(ctrl)
    z = (abs(rho_p) - np.abs(ctrl).mean()) / (np.abs(ctrl).std(ddof=1) + 1e-18)
    print(f'[{name:<12}] 단독BSS={sc(p):8.2f}  직교후rho={rho_p:+.5f}  '
          f'이득={K*Ap**2/Vp if Vp>1e-18 else 0:+.2f}  s*={-Ap/Vp if Vp>1e-18 else 0:+.4f}  '
          f'z={z:5.1f}  {"통과" if z>2 else "허수"}', flush=True)
    return z


results = {}
for split_name, bucket in SPLITS.items():
    log(f'=== {split_name} 분할 실험 ===')
    p_comp = np.zeros(va.sum())
    ok = True
    for b in np.unique(bucket):
        tr = tr_era & (cls >= 0) & (bucket == b)
        if tr.sum() < 50000:
            log(f'  버킷{b} 행수 부족({tr.sum():,}) - 이 분할 스킵')
            ok = False
            break
        w = 0.5 ** ((2023.0 - season[tr]) / 2.0)
        Xtr, ctr = X.loc[tr], cls[tr]
        n_es = int(len(Xtr) * 0.92)
        log(f'  버킷{b} 학습행 {tr.sum():,}')
        m = CatBoostClassifier(**CB)
        m.fit(Xtr.iloc[:n_es], ctr[:n_es], sample_weight=w[:n_es],
              eval_set=(Xtr.iloc[n_es:], ctr[n_es:]))
        mask_va = bucket[va] == b
        proba = m.predict_proba(X.loc[va][mask_va])
        p_comp[mask_va] = np.clip(proba[:, SUCC].sum(axis=1), 0, 1)
        log(f'  버킷{b} 완료 best_iter={m.best_iteration_}')
    if not ok:
        continue
    np.save(f'dev/cache_split_{split_name}_A.npy', p_comp)
    results[split_name] = screen(split_name, p_comp)

print('\n=== 종합 ===')
for nm, z in results.items():
    print(f'  {nm}: z={z:.1f}  {"통과" if z > 2 else "허수"}')
log('전체 완료')
