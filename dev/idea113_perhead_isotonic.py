"""헤드별 isotonic 재보정 - 각 헤드가 0.5 근처로 눌려있다면(랭킹은 맞는데 확신 부족)
isotonic이 실제 관측성공률에 맞게 펴준다. H1/H2 양방향으로 fit->eval, raw 대비 이득 확인."""
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
va = season == 2024
yv = y[va]

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
avg = lambda ps: np.mean([np.load(p) for p in ps], axis=0)
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
P11 = np.load('dev/idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)

keys = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50', 'mc5', 'ingame']
W = {k: v88[f'{k}_weight'] for k in keys}
raw = sum(W[k] * H[k] for k in keys)
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
v88_final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

X_ = X.loc[va]
mth = X_['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
sc = lambda p_, m: 1e5 * (1 - np.mean((np.clip(p_[m], 0, 1) - yv[m]) ** 2) / unc)

print(f'raw v88_final 전체 = {sc(v88_final, np.ones(len(yv),bool)):.2f}\n')

# 헤드별 압축도(표준편차) 확인 - 압축이 심한 애들이 isotonic 효과 클 가능성
print('=== 헤드별 표준편차(압축도 참고) ===')
for k in keys:
    print(f'  {k:12s} std={H[k].std():.4f}  range=[{H[k].min():.3f},{H[k].max():.3f}]')

def calib_headwise(fit_m, ev_m):
    H_calib = {}
    for k in keys:
        iso = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        iso.fit(H[k][fit_m], yv[fit_m])
        h_ev = H[k].copy()
        h_ev[ev_m] = iso.transform(H[k][ev_m])
        H_calib[k] = h_ev
    raw_c = sum(W[k] * H_calib[k] for k in keys)
    v_c = np.clip(raw_c - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)
    return v_c

print(f'\n=== 헤드별 isotonic 재보정 후 (기존 선형가중치 그대로 사용) ===')
gains = []
for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
    v_c = calib_headwise(fit_m, ev_m)
    g_raw = sc(v88_final, ev_m)
    g_calib = sc(v_c, ev_m)
    gains.append(g_calib - g_raw)
    print(f'{tag}: raw={g_raw:.2f}  isotonic보정후={g_calib:.2f}  이득={g_calib-g_raw:+.2f}')
print(f'평균 이득 = {np.mean(gains):+.2f}')
log('완료')
