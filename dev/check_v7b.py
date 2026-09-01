import sys, importlib.util
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

spec = importlib.util.spec_from_file_location("script", "submit/script.py")
script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(script)

a = joblib.load('dev/old_models/model_artifacts_v7b.pkl')
print('keys:', sorted(a.keys()))
print('feature_order 길이:', len(a['feature_order']))
print(a['feature_order'][:20])

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()

stats = a['stats']
X_base = script.build_features(df, stats)
X_ins = script.build_inseason_features(df, a['inseason_stats'])
prior_rate = script.get_prior_pitcher_rate(df, a['inseason_stats'])
X_plt = script.build_platoon_features(df, a['platoon_stats'], prior_rate)

X = pd.concat([X_base, X_ins, X_plt], axis=1)
X = X[a['feature_order']].astype(np.float64)
print('피처 구성 완료:', X.shape)

hgb = a['hgb']
pred = np.clip(hgb.predict_proba(X)[:, 1] if hasattr(hgb, 'predict_proba') else hgb.predict(X), 0, 1)

unc = 0.249807
for s in (2022, 2023, 2024):
    m = season == s
    yv = y[m]; pv = pred[m]
    bss = 1e5 * (1 - np.mean((pv - yv) ** 2) / unc)
    print(f'season={s}  n={m.sum():,}  BSS={bss:.1f}  실제평균={yv.mean():.4f}  예측평균={pv.mean():.4f}')

# 전체(2019-2024)에도 재보기 - in-sample 여부 확인용
bss_all = 1e5 * (1 - np.mean((pred - y) ** 2) / unc)
print(f'전체(2019-2024) BSS={bss_all:.1f}')
