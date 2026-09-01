"""잔차 헤드: 현재 v117 블렌드가 '무엇을 틀리는지'를 162피처로 직접 학습.

지금까지 헤드는 전부 '어떤 분해가 도움될까'를 추측해서 만든 것.
이건 반대 방향 - 블렌드의 잔차(y - blend)를 타겟으로 비선형 모델을 학습한다.

선형 스태킹(실패)과 다른 점: 스태킹은 헤드 출력의 재가중(선형). 이건 원본 162피처로
비선형 CatBoost가 잔차를 직접 학습 -> 완전히 다른 함수공간.

검증(교차연도 이식, stacking-closed-and-crossfold-method 방식):
  fold A 잔차로 학습 -> fold C에서 평가
  fold C 잔차로 학습 -> fold A에서 평가
양방향 모두 양수여야 진짜 신호. 한쪽만 좋으면 그 해 특유 노이즈.
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
v117 = joblib.load('submit/model/model_artifacts_v117.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

# v117 구조의 블렌드를 fold A/C에서 재현 (mc6=0.48, strk=0.10, 나머지 v95 x 0.42)
S_MC6, S_STRK = 0.48, 0.10
HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']


def build_v117_blend(tag):
    H = dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )
    W = {k: float(v95[f'{k}_weight']) for k in HEADS8}
    t_ = sum(W.values())
    rest = 1.0 - S_MC6 - S_STRK
    base8 = sum((W[k] / t_) * H[k] for k in HEADS8)
    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    return np.clip(rest * base8 + S_MC6 * p_mc6 + S_STRK * p_strk, 0, 1)


FOLDS = {'A': 2024, 'C': 2022}
blends, resids, masks = {}, {}, {}
for tag, vs in FOLDS.items():
    m = season == vs
    b = build_v117_blend(tag)
    blends[tag] = b
    resids[tag] = y[m] - b
    masks[tag] = m
    sc = 1e5 * (1 - np.mean((b - y[m]) ** 2) / B_)
    log(f'fold {tag}({vs}) v117구조 블렌드 BSS = {sc:.1f}  잔차 sd={resids[tag].std():.5f}')

CAT = dict(iterations=1000, learning_rate=0.03, depth=6, l2_leaf_reg=8.0, verbose=0,
           loss_function='RMSE', early_stopping_rounds=50, random_seed=42)

print('\n' + '=' * 78)
print('교차연도 검증: 한 fold 잔차로 학습 -> 다른 fold에서 평가')
print('=' * 78)
results = {}
for tr_tag, ev_tag in [('A', 'C'), ('C', 'A')]:
    m_tr, m_ev = masks[tr_tag], masks[ev_tag]
    Xtr, rtr = X.loc[m_tr], resids[tr_tag]
    n_es = int(len(Xtr) * 0.90)
    ts = time.time()
    mdl = CatBoostRegressor(**CAT)
    mdl.fit(Xtr.iloc[:n_es], rtr[:n_es],
            eval_set=(Xtr.iloc[n_es:], rtr[n_es:]))
    rhat = mdl.predict(X.loc[m_ev])
    log(f'[{tr_tag}->{ev_tag}] 학습완료 best_iter={mdl.best_iteration_} ({time.time()-ts:.0f}s)')
    np.save(f'dev/cache_residhead_{tr_tag}to{ev_tag}.npy', rhat)

    yv = y[m_ev]
    b = blends[ev_tag]
    r_true = resids[ev_tag]
    d = rhat - rhat.mean()
    V = float(np.mean(d ** 2))
    A = float(np.mean(d * (b - yv)))
    rho = -A / np.sqrt(V * float(np.mean(r_true ** 2))) if V > 1e-14 else 0.0
    gain = K * A ** 2 / V if V > 1e-14 else 0.0
    s_opt = -A / V if V > 1e-14 else 0.0
    corr_rr = float(np.corrcoef(rhat, r_true)[0, 1])
    results[(tr_tag, ev_tag)] = (rho, gain, s_opt, corr_rr)
    print(f'  {tr_tag}->{ev_tag}: rhat_sd={rhat.std():.5f}  corr(rhat, 실제잔차)={corr_rr:+.5f}')
    print(f'         rho={rho:+.5f} (필요치의 {abs(rho)/NEED_RHO*100:.1f}%)  '
          f's*={s_opt:+.4f}  최대이득(로컬)={gain:+.2f}')

print('\n' + '=' * 78)
print('[판정]')
print('=' * 78)
signs = [np.sign(v[0]) for v in results.values()]
agree = len(set(signs)) == 1
print(f'  양방향 부호일치: {"O" if agree else "X"}')
for (a, b_), (rho, gain, s_opt, cr) in results.items():
    print(f'  {a}->{b_}: rho={rho:+.5f}  corr(rhat,잔차)={cr:+.5f}  최대이득={gain:+.2f}')
print('\n  corr(rhat, 실제잔차)가 양쪽 모두 유의하게 양수여야 "블렌드 오차를 실제로 예측"한 것.')
print('  0 근처면 잔차가 예측 불가능한 노이즈라는 뜻 -> 이 축도 닫힘.')
log('완료')
