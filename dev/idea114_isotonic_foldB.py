"""fold B(train<=2022 -> 2023, '지랄맞은' 해)로 isotonic 재보정 재검증.
사용가능 헤드만(base/hurdle/multires/ordinal) 비례가중치로 blend 구성 후,
H1/H2(2023 상반기/하반기)로 isotonic 검증."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.isotonic import IsotonicRegression

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
unc = 0.249807
va = season == 2023
yv = y[va]
log(f'fold B eval(2023) n={va.sum():,}')

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
avg = lambda ps: np.mean([np.load(p) for p in ps], axis=0)
p = 'B'
H = dict(
    base=avg([f'dev/phase90_cache/{p}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
    hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{p}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
    multires=avg([f'dev/idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'dev/idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
)
keys = list(H.keys())
W0 = {k: v88[f'{k}_weight'] for k in keys}
tot_w = sum(W0.values())
W = {k: W0[k] / tot_w for k in keys}  # 4개만으로 비례 재정규화(근사)
print('사용 가중치(재정규화):', W)

v_blend = sum(W[k] * H[k] for k in keys)
X_ = X.loc[va]
mth = X_['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
sc = lambda p_, m: 1e5 * (1 - np.mean((np.clip(p_[m], 0, 1) - yv[m]) ** 2) / unc)
allm = np.ones(len(yv), bool)
print(f'\nraw blend(4헤드,근사) 전체 = {sc(v_blend, allm):.2f}')

def calib_headwise(fit_m, ev_m):
    H_calib = {}
    for k in keys:
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(H[k][fit_m], yv[fit_m])
        h_ev = H[k].copy()
        h_ev[ev_m] = iso.transform(H[k][ev_m])
        H_calib[k] = h_ev
    return sum(W[k] * H_calib[k] for k in keys)

print(f'\n=== fold B(2023) 헤드별 isotonic 재보정 검증 ===')
gains = []
for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
    v_c = calib_headwise(fit_m, ev_m)
    g_raw = sc(v_blend, ev_m)
    g_calib = sc(v_c, ev_m)
    gains.append(g_calib - g_raw)
    print(f'{tag}: raw={g_raw:.2f}  isotonic보정후={g_calib:.2f}  이득={g_calib-g_raw:+.2f}')
print(f'평균 이득 = {np.mean(gains):+.2f}')
log('완료')
