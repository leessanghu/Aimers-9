"""mc5(11-class, 최근 최대 기여 헤드)를 시드만 바꿔 여러 번 학습 -> 평균(시드배깅).
정직 fold A(train<=2023 -> 2024)에서 단일시드 vs 배깅 BSS 비교.
v88 프로덕션과 동일 config: depth=6, iterations=1000, lr=0.05, l2=5.0, early_stopping=40."""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy(np.float64)
cls5 = np.load('dev/cls5_labels.npy')
pt = np.load('dev/pitchtype_labels.npy')
unc = 0.249807

v = (cls5 >= 0) & (pt >= 0)
cls = np.full(len(cls5), -1, dtype=np.int64)
nd = v & (cls5 >= 2); cls[nd] = (cls5[nd] - 2) * 3 + pt[nd]
cls[v & (cls5 == 0)] = 9; cls[v & (cls5 == 1)] = 10

tr = season <= 2023; va = season == 2024
fit = tr & (cls >= 0)
w = 0.5 ** ((2023 - season) / 2.0)
fi = np.where(fit)[0]; n_es = int(len(fi) * 0.92)
ti, ei = fi[:n_es], fi[n_es:]
log(f'학습행 {fit.sum():,}  fold A eval {va.sum():,}')

os.makedirs('dev/idea102_cache', exist_ok=True)
SEEDS = [42, 7, 123]
Ps = []
for sd in SEEDS:
    fcache = f'dev/idea102_cache/A_proba11_seed{sd}.npy'
    if os.path.exists(fcache):
        P = np.load(fcache); log(f'seed{sd} 캐시 사용')
    else:
        m = CatBoostClassifier(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
                                verbose=False, random_seed=sd, loss_function='MultiClass',
                                classes_count=11, early_stopping_rounds=40)
        m.fit(X.iloc[ti], cls[ti], sample_weight=w[ti], eval_set=(X.iloc[ei], cls[ei]))
        log(f'seed{sd} 학습완료 best_iter={m.best_iteration_}')
        P = m.predict_proba(X.loc[va])
        np.save(fcache, P)
    Ps.append(P)

# v88_final 재구성 (idea91과 동일, mc5만 교체)
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
p = 'A'
H = dict(
    base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'dev/idea46_cache/{p}_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'dev/idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'dev/idea54_cache/{p}_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'dev/idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42, 7)]),
)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)

yv = y[va]
sc = lambda p_: 1e5 * (1 - np.mean((np.clip(p_, 0, 1) - yv) ** 2) / unc)

mc5_succ = np.asarray(v88['mc5_succ'], dtype=np.float64)

def build_v88_final(P11):
    H2 = dict(H); H2['mc5'] = np.clip(P11 @ mc5_succ, 0, 1)
    raw = sum(W[k] * H2[k] for k in H2)
    risk = P11[:, [9, 10]].sum(axis=1)
    cut = np.maximum(0.0, risk - float(v88['risk_thr']))
    return np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

print()
print(f'=== 단일시드 vs 배깅 (fold A, honest) ===')
for i, sd in enumerate(SEEDS):
    s = sc(build_v88_final(Ps[i]))
    print(f'  seed{sd} 단독            v88_final={s:.2f}')

P2 = np.mean(Ps[:2], axis=0)
s2 = sc(build_v88_final(P2))
print(f'  2-seed 배깅(42,7)        v88_final={s2:.2f}')

P3 = np.mean(Ps, axis=0)
s3 = sc(build_v88_final(P3))
print(f'  3-seed 배깅(42,7,123)    v88_final={s3:.2f}')

base_single = sc(build_v88_final(Ps[0]))
print(f'\n델타(2-seed - seed42단독)  = {s2 - base_single:+.2f}')
print(f'델타(3-seed - seed42단독)  = {s3 - base_single:+.2f}')

# proba11 자체의 분산(신뢰클래스 9,10 확률의 시드간 표준편차)도 참고로 확인
p9_std = np.std([P[:, 9] for P in Ps], axis=0).mean()
p10_std = np.std([P[:, 10] for P in Ps], axis=0).mean()
print(f'\n클래스9(middle) 확률 시드간 평균std = {p9_std:.5f}')
print(f'클래스10(reverse) 확률 시드간 평균std = {p10_std:.5f}')
log('완료')
