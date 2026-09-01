"""mc5 가중치 상향 + base/multires 하향 재배분. 캐시된 fold A 헤드예측 재사용(재학습 없음).
주의: 로컬은 구조적으로 aux(mc5)를 저평가/base를 과대평가하는 편향이 있음(기존 확립됨).
그래도 방향성 체크 + H1/H2로 참고용 확인."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

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
P11 = np.load('dev/idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)

keys = ['base', 'hurdle', 'multires', 'ordinal', 'midother', 'condball', 'countresid', 'future50', 'mc5', 'ingame']
W_old = {k: v88[f'{k}_weight'] for k in keys}

def build(Wd):
    raw = sum(Wd[k] * H[k] for k in keys)
    risk = P11[:, [9, 10]].sum(axis=1)
    cut = np.maximum(0.0, risk - float(v88['risk_thr']))
    return np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

sc = lambda p_, m: 1e5 * (1 - np.mean((np.clip(p_[m], 0, 1) - yv[m]) ** 2) / unc)
allm = np.ones(len(yv), bool)

pred_old = build(W_old)
print(f'기존 가중치  v88_final = {sc(pred_old, allm):.2f}')

W_new = dict(W_old)
W_new['base'] -= 0.02
W_new['multires'] -= 0.015
W_new['mc5'] += 0.035
assert abs(sum(W_new.values()) - 1.0) < 1e-9
pred_new = build(W_new)
print(f'신규 가중치(안1)  v88_final = {sc(pred_new, allm):.2f}   (base-0.02, multires-0.015, mc5+0.035 -> {W_new["mc5"]:.3f})')
print(f'델타 = {sc(pred_new, allm) - sc(pred_old, allm):+.2f}')

W_new2 = dict(W_old)
W_new2['base'] -= 0.02
W_new2['multires'] -= 0.015
W_new2['condball'] -= 0.0085
W_new2['countresid'] -= 0.0085
W_new2['mc5'] += 0.052
assert abs(sum(W_new2.values()) - 1.0) < 1e-9
pred_new2 = build(W_new2)
print(f'신규 가중치(안2, mc5=hurdle급)  v88_final = {sc(pred_new2, allm):.2f}   (mc5 -> {W_new2["mc5"]:.3f}, hurdle={W_new2["hurdle"]:.3f})')
print(f'델타 = {sc(pred_new2, allm) - sc(pred_old, allm):+.2f}')

X_ = X.loc[va]
mth = X_['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
print()
for m, tag in [(H1, 'H1(1-6월)'), (H2, 'H2(7-12월)')]:
    d_old = sc(pred_old, m); d_new = sc(pred_new, m)
    print(f'  {tag}: 기존={d_old:.2f}  신규={d_new:.2f}  델타={d_new-d_old:+.2f}')
log('완료')
