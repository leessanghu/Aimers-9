"""v7c, v9 (진짜 단일모델, w_hgb=1.0) 정직 재학습 + v88_final과 오차상관/블렌드이득."""
import sys, importlib.util, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

spec = importlib.util.spec_from_file_location("script", "submit/script.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
unc = 0.249807

# v88_final 준비 (공통)
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
va = season == 2024
yv = y[va]
v88_final = np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)
sc = lambda p_: 1e5 * (1 - np.mean((np.clip(p_, 0, 1) - yv) ** 2) / unc)
resid_v88 = yv - v88_final
print(f'v88_final 단독 BSS = {sc(v88_final):.1f}')
print()


def eval_old(name, path):
    log(f'=== {name} ===')
    a = joblib.load(path)
    feats = [script.build_features(df, a['stats'])]
    prior_rate = None
    if 'inseason_stats' in a:
        feats.append(script.build_inseason_features(df, a['inseason_stats']))
        prior_rate = script.get_prior_pitcher_rate(df, a['inseason_stats'])
    if 'platoon_stats' in a:
        feats.append(script.build_platoon_features(df, a['platoon_stats'], prior_rate))
    if 'inning_stats' in a:
        feats.append(script.build_inning_features(df, a['inning_stats'], prior_rate))
    X = pd.concat(feats, axis=1)
    missing = [c for c in a['feature_order'] if c not in X.columns]
    if missing:
        print(f'  {name}: 스킵 - build_features로 못 만드는 피처 {len(missing)}개 (예: {missing[:5]})')
        return None
    X = X[a['feature_order']].astype(np.float64)
    tr = season <= 2023
    Xt, yt = X.loc[tr], y[tr]
    Xv = X.loc[va]
    hgb = HistGradientBoostingClassifier(
        max_iter=500, learning_rate=0.03, max_depth=6, max_leaf_nodes=31,
        l2_regularization=5.0, min_samples_leaf=20, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=20, random_state=42)
    hgb.fit(Xt, yt)
    pred = np.clip(hgb.predict_proba(Xv)[:, 1], 0, 1)
    solo = sc(pred)
    err = pred - yv
    rho = np.corrcoef(err, resid_v88 * -1)[0, 1]  # resid_v88 = y - v88_final, err_v88 = -resid_v88
    B1 = np.mean((v88_final - yv) ** 2); B2 = np.mean(err ** 2)
    thresh = np.sqrt(B1 / B2)
    d = pred - v88_final
    C = np.mean(d * resid_v88); V = np.mean(d ** 2)
    a_star = C / V if V > 1e-12 else 0.0
    blend = v88_final + a_star * d
    print(f'  {name}: solo={solo:.1f}  n_iter={hgb.n_iter_}  err_corr={rho:.4f}  '
          f'임계={thresh:.4f}  {"통과" if rho<thresh else "미달"}  a*={a_star:.4f}  '
          f'블렌드={sc(blend):.1f} ({sc(blend)-sc(v88_final):+.2f})')
    return pred


preds = {}
for name, path in [('v7c', 'dev/old_models/model_artifacts_v7c.pkl'),
                    ('v9', 'dev/old_models/model_artifacts_v9.pkl')]:
    preds[name] = eval_old(name, path)
