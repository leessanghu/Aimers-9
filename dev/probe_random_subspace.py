"""H2/H3 처방 검증: '피처가 비슷해서 헤드들이 수렴한다(실효랭크 1.1/8)'면
피처 부분공간을 강제로 쪼개면 다양성이 늘어 앙상블이 이득인가?

정확한 비교(핵심): 같은 개수의 모델을
  (A) 전체 162피처 + 시드만 다르게      <- 대조군 ("모델 더 쌓기")
  (B) 랜덤 60% 피처 부분공간            <- 처방 ("강제 다양화")
로 학습해서 ambiguity decomposition으로 분해한다.

  BS_ens = mean_i BS_i - Ambiguity
(B)가 이기려면: ambiguity 증가분 > 개별오차 증가분. 이게 이 처방의 성패 전부다.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807
K_C = 1e5 / B
NMODEL = 5
SUBSPACE_FRAC = 0.60

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64)

PARAMS = dict(iterations=800, learning_rate=0.04, depth=6, l2_leaf_reg=5.0,
              verbose=0, early_stopping_rounds=50, loss_function='Logloss')


def run_fold(upto, vs, tag):
    tr = season <= upto
    va = season == vs
    Xtr_all, ytr = X.loc[tr], y[tr]
    Xva_all, yva = X.loc[va], y[va]
    w = 0.5 ** ((upto - season[tr]) / 2.0)
    n_es = int(len(Xtr_all) * 0.92)

    bs = lambda p: float(np.mean((np.clip(p, 0, 1) - yva) ** 2))
    sc = lambda p: 1e5 * (1 - bs(p) / B)

    results = {}
    for mode in ('full', 'subspace'):
        preds = []
        for i in range(NMODEL):
            rng = np.random.RandomState(1000 + i)
            if mode == 'full':
                cols = FEAT
            else:
                cols = sorted(rng.choice(FEAT, size=int(len(FEAT) * SUBSPACE_FRAC),
                                         replace=False).tolist())
            Xtr = Xtr_all[cols]; Xva = Xva_all[cols]
            ts = time.time()
            m = CatBoostClassifier(**PARAMS, random_seed=1000 + i)
            m.fit(Xtr.iloc[:n_es], ytr[:n_es], sample_weight=w[:n_es],
                  eval_set=(Xtr.iloc[n_es:], ytr[n_es:]))
            p = np.clip(m.predict_proba(Xva)[:, 1], 0, 1)
            preds.append(p)
            log(f'  [{tag}/{mode}] model{i} feats={len(cols)} BSS={sc(p):7.1f} '
                f'best_iter={m.best_iteration_} ({time.time()-ts:.0f}s)')
        P = np.column_stack(preds)
        fbar = P.mean(axis=1)
        mean_indiv = float(np.mean([bs(P[:, i]) for i in range(NMODEL)]))
        ambig = float(np.mean([np.mean((P[:, i] - fbar) ** 2) for i in range(NMODEL)]))
        Cm = np.corrcoef(P.T)
        off = Cm[np.triu_indices(NMODEL, 1)]
        results[mode] = dict(mean_indiv=mean_indiv, ambig=ambig, ens=bs(fbar),
                             corr=off.mean(), sc_ens=sc(fbar),
                             sc_indiv=1e5 * (1 - mean_indiv / B))

    print(f'\n=== fold {tag} ({vs}) 비교 ===')
    print(f'{"":<12}{"개별평균BSS":>13}{"ambiguity":>13}{"->점수":>9}{"앙상블BSS":>12}{"모델간상관":>11}')
    for mode in ('full', 'subspace'):
        r = results[mode]
        print(f'{mode:<12}{r["sc_indiv"]:>13.1f}{r["ambig"]:>13.6f}'
              f'{K_C*r["ambig"]:>+9.1f}{r["sc_ens"]:>12.1f}{r["corr"]:>11.4f}')
    d_ind = results['subspace']['sc_indiv'] - results['full']['sc_indiv']
    d_amb = K_C * (results['subspace']['ambig'] - results['full']['ambig'])
    d_ens = results['subspace']['sc_ens'] - results['full']['sc_ens']
    print(f'\n  강제다양화 효과 분해:')
    print(f'    개별오차 변화  = {d_ind:+.1f}점  (음수면 개별모델이 나빠진 것)')
    print(f'    ambiguity 변화 = {d_amb:+.1f}점  (양수면 다양성 증가)')
    print(f'    순 앙상블 효과 = {d_ens:+.1f}점  <- 이게 최종 판정')
    return results


log('=== fold A (train<=2023 -> 2024) ===')
run_fold(2023, 2024, 'A')
log('=== fold C (train<=2021 -> 2022) ===')
run_fold(2021, 2022, 'C')
log('완료')
