"""우리 블렌드(v117)가 틀리는데 XGB/LGBM이 맞는 지점이 실제로 있는지 진단.
스칼라 집계(잔차상관) 대신, row-level advantage를 타겟으로 삼아
얕은 HGB로 '어떤 피처 조합에서 XGB가 이기는지' 예측 가능한지 검사.
예측 가능하면(AUC 유의하게 >0.5, fold간 재현) 그 세그먼트를 피처중요도로 찾아냄.
예측 불가능하면(랜덤 수준) '어디서 XGB가 이기는지' 자체가 노이즈라는 뜻.
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B_ = 0.249807
X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT = list(v95['feature_order'])
X = X[FEAT].astype(np.float64).fillna(0.0)
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

HEADS8 = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
          'condball', 'countresid', 'future50']
S_MC6, S_STRK = 0.48, 0.10


def build8(tag):
    return dict(
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


HGB = dict(max_depth=5, max_leaf_nodes=31, max_iter=250, learning_rate=0.05,
           l2_regularization=5.0, early_stopping=True, validation_fraction=0.1,
           n_iter_no_change=20, random_state=42)

for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y_all[va]
    Xv = X.loc[va]
    H = build8(tag)
    W0 = {k: float(v95[f'{k}_weight']) for k in HEADS8}
    t_ = sum(W0.values()); W0 = {k: v / t_ for k, v in W0.items()}
    blend8 = np.clip(sum(W0[k] * H[k] for k in HEADS8), 0, 1)
    p_mc6 = np.load(f'dev/cache_mc6head_{tag}.npy')
    p_strk = np.load(f'dev/cache_strk_strk_linear_{tag}.npy')
    rest = 1.0 - S_MC6 - S_STRK
    blend = np.clip(rest * blend8 + S_MC6 * p_mc6 + S_STRK * p_strk, 0, 1)

    print(f'\n{"="*80}\n=== fold {tag} ({vs}) ===\n{"="*80}')
    for model_name in ['xgb_rawid', 'lgbm_rawid']:
        p_alt = np.load(f'dev/cache_{model_name.replace("_","")}_{tag}.npy')
        e_blend = (blend - yv) ** 2
        e_alt = (p_alt - yv) ** 2
        win = (e_alt < e_blend).astype(np.int64)   # 1 = XGB/LGBM이 그 행에서 더 정확
        win_rate = win.mean()
        # BS 단위로 '이겼을 때 얼마나 이겼는지' 도 참고
        adv = e_blend - e_alt   # 양수=XGB가 나음
        print(f'\n--- {model_name} ---')
        print(f'  XGB/LGBM이 더 정확한 행 비율: {win_rate*100:.2f}%  (50%면 순수동전던지기)')
        print(f'  평균 advantage(양수=XGB나음) = {adv.mean():+.6e}')

        n_es = int(len(Xv) * 0.9)
        idx = np.random.RandomState(42).permutation(len(Xv))
        tr_idx, va_idx = idx[:n_es], idx[n_es:]
        clf = HistGradientBoostingClassifier(**HGB)
        clf.fit(Xv.iloc[tr_idx], win[tr_idx])
        p_win = clf.predict_proba(Xv.iloc[va_idx])[:, 1]
        try:
            auc = roc_auc_score(win[va_idx], p_win)
        except ValueError:
            auc = float('nan')
        print(f'  "이 행에서 XGB가 이길지" 예측가능한가 (HGB val AUC) = {auc:.4f}  (0.50=예측불가/노이즈)')

        if auc > 0.55:
            imp_idx = np.argsort(clf.feature_importances_ if hasattr(clf, 'feature_importances_') else [])[::-1]
            # HGB엔 feature_importances_가 없으므로 permutation 생략, 대신 상위 advantage 세그먼트 직접 탐색
        # advantage와 각 피처의 단순 상관(선형, 참고용) - 상위 몇 개만
        advc = adv - adv.mean()
        corrs = []
        for c in FEAT:
            xv = Xv[c].to_numpy()
            xvc = xv - xv.mean()
            denom = np.sqrt(np.mean(xvc**2) * np.mean(advc**2)) + 1e-18
            if denom < 1e-12:
                continue
            corrs.append((c, float(np.mean(xvc*advc) / denom)))
        corrs.sort(key=lambda t: -abs(t[1]))
        print(f'  advantage와 상관 top8 피처(참고용, 선형상관):')
        for c, r in corrs[:8]:
            print(f'    {c:<40} corr={r:+.4f}')
