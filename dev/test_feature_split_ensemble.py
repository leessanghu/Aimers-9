"""162피처를 중요도 하위 40 vs 상위 122로 쪼개서, 하위40 전용 모델이 v88_final 잔차와
진짜 상관이 있는지(=최대이득 공식) 확인. fold A(train<=2023->2024) 정직검증."""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostRegressor

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807
K = 1e5 / B

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
feats = v88['feature_order']

# 전체 헤드(catboost 계열) 평균 중요도로 랭킹
model_keys = ['mc5_model', 'midother_model', 'condball_model', 'countresid_model', 'future50_model', 'ingame_model']
agg = np.zeros(len(feats))
cnt = 0
for mk in model_keys:
    m = v88.get(mk)
    if m is None:
        continue
    agg += np.array(m.get_feature_importance())
    cnt += 1
avgimp = agg / max(cnt, 1)
order = np.argsort(avgimp)  # 오름차순: 낮은 중요도부터
bottom40 = [feats[i] for i in order[:40]]
top122 = [feats[i] for i in order[40:]]
log(f'하위40 예시: {bottom40[:10]}')
log(f'하위40 총 중요도합={avgimp[order[:40]].sum():.2f}  전체합={avgimp.sum():.2f}  ({avgimp[order[:40]].sum()/avgimp.sum()*100:.1f}%)')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
tr = season <= 2023
va = season == 2024
yv = y[va]
w = 0.5 ** ((2023 - season) / 2.0)
ti_all = np.where(tr)[0]
n_es = int(len(ti_all) * 0.92)
ti, ei = ti_all[:n_es], ti_all[n_es:]

CFG = dict(iterations=700, learning_rate=0.03, depth=6, l2_leaf_reg=5.0, verbose=0,
           loss_function='RMSE', early_stopping_rounds=40)

def train(feat_list, seed):
    m = CatBoostRegressor(**CFG, random_seed=seed)
    m.fit(X.iloc[ti][feat_list], y[ti], sample_weight=w[ti],
          eval_set=(X.iloc[ei][feat_list], y[ei]))
    return np.clip(m.predict(X.loc[va, feat_list]), 0, 1)

log('하위40 모델 학습...')
p_b40 = train(bottom40, 61)
log('상위122 모델 학습...')
p_t122 = train(top122, 62)

base = np.load('dev/cache_v88_final_2024.npy')
resid = yv - base
sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)

X_ = X.loc[va]
mth = X_['game_month'].to_numpy()
H1 = mth <= 6
H2 = ~H1


def analyze(p, tag):
    g = p - p.mean()
    C = np.mean(g * resid)
    Eg2 = np.mean(g ** 2)
    rho = C / np.sqrt(Eg2 * resid.var())
    max_gain = (C ** 2 / Eg2) * K
    print(f'\n--- {tag} ---')
    print(f'  단독 BSS = {sc(p, np.ones(len(yv), bool)):.2f}')
    print(f'  corr(g, resid) = {rho:+.5f}   (+20점 문턱 대비 {abs(rho)/0.0142*100:.1f}%)')
    print(f'  이론 최대이득(fold A 전체) = {max_gain:+.2f}점')
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        Cf = np.mean(g[fit_m] * resid[fit_m])
        Vf = np.mean(g[fit_m] ** 2)
        a = Cf / Vf if Vf > 1e-12 else 0.0
        bl = base.copy()
        bl[ev_m] = base[ev_m] + a * g[ev_m]
        gains.append(sc(bl, ev_m) - sc(base, ev_m))
    print(f'  H1->H2 이득={gains[0]:+.2f}  H2->H1 이득={gains[1]:+.2f}  평균={np.mean(gains):+.2f}')


analyze(p_b40, '하위40 전용 모델')
analyze(p_t122, '상위122 전용 모델(참고)')

# 두 모델 서로간, 그리고 v88_final과의 오차상관
r_b40 = yv - p_b40
r_full = resid
ecorr = np.corrcoef(r_b40, r_full)[0, 1]
print(f'\n하위40모델 vs v88_final 오차상관 = {ecorr:.4f}  (참고: 오늘 이질모델들은 0.995~0.999)')
log('완료')
