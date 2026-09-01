"""팀원 모델(v1_inseason_resid) 검증: 2024 실제 예측 BSS(in-sample 여부 확인) +
v88_final과의 오차상관/블렌드이득."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, 'dev/teammate_v1')
import numpy as np, pandas as pd, joblib
import pipeline as P

art = joblib.load('dev/teammate_v1/model/model_artifacts_v1_inseason_resid.pkl')
print('version:', art['version'], ' val_score(보고값):', art['val_score'], ' wr:', art['wr'])

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
unc = 0.249807

tm = P._load_tm()
df_fe = P.apply_fe(df, tm, art['tables'])

feats, cats = art['feats'], art['cats']
rfeats, cats_r = art['rfeats'], art['cats_r']


def fill(d, cols, cc):
    d = d[cols].copy()
    for c in cc:
        if c in d.columns:
            d[c] = d[c].fillna('missing')
    return d


p_base = art['base_model'].predict_proba(fill(df_fe, feats, cats))[:, 1]
base_ins = df_fe['inseason_success'].fillna(0.5).values
p_resid = np.clip(base_ins + art['resid_model'].predict(fill(df_fe, rfeats, cats_r)), 0, 1)
pt = (1 - art['wr']) * p_base + art['wr'] * p_resid

eps = 1e-6
lg = np.log(np.clip(pt, eps, 1 - eps) / (1 - np.clip(pt, eps, 1 - eps)))
Xc = np.column_stack([lg ** k for k in range(1, 6)])
z = Xc @ np.asarray(art['calib_coef']).ravel() + art['calib_intercept']
pred = np.clip(1 / (1 + np.exp(-z)), 1e-4, 1 - 1e-4)

print()
for s in (2020, 2021, 2022, 2023, 2024):
    m = season == s
    yv = y[m]; pv = pred[m]
    bss = 1e5 * (1 - np.mean((pv - yv) ** 2) / unc)
    print(f'season={s}  n={m.sum():,}  BSS={bss:.1f}  실제={yv.mean():.4f}  예측={pv.mean():.4f}')

va = season == 2024
yv = y[va]
sc = lambda p_: 1e5 * (1 - np.mean((np.clip(p_, 0, 1) - yv) ** 2) / unc)

# v88_final 준비
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
P11 = np.load('dev/idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('dev/idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
raw = sum(W[k] * H[k] for k in H)
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))
v88_final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

pred_v = pred[va]
print()
print(f'v88_final BSS = {sc(v88_final):.1f}')
print(f'teammate(2024) BSS = {sc(pred_v):.1f}')
err_t = pred_v - yv
err_v88 = v88_final - yv
rho = np.corrcoef(err_t, err_v88)[0, 1]
pred_rho = np.corrcoef(pred_v, v88_final)[0, 1]
print(f'corr(teammate_err, v88_err) = {rho:.4f}')
print(f'corr(teammate_pred, v88_pred) = {pred_rho:.4f}')

B1 = np.mean(err_v88 ** 2); B2 = np.mean(err_t ** 2)
thresh = np.sqrt(B1 / B2)
print(f'B1={B1:.6f}  B2={B2:.6f}  임계={thresh:.6f}  {"통과" if rho<thresh else "미달"}')

resid_v88 = yv - v88_final
d = pred_v - v88_final
C = np.mean(d * resid_v88); V = np.mean(d ** 2)
a_star = C / V if V > 1e-12 else 0.0
blend = v88_final + a_star * d
print(f'최적계수 a*={a_star:.4f}  블렌드 BSS={sc(blend):.1f}  (v88단독 대비 {sc(blend)-sc(v88_final):+.2f})')
