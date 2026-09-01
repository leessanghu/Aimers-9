"""TabM 재현성 검증: fold C(train<=2021 -> 2022)에서도 같은 기여가 나오는가.
+ 배포 가능성 점검(로컬 GPU/torch/tabm)."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


df_raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['row_id', 'season'])

for tag, vs in [('C', 2022), ('A', 2024)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v88[f'{k}_weight']) for k in H}
    tot = sum(W.values())
    W = {k: v / tot for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / B)
    mth = X.loc[va, 'game_month'].to_numpy()
    H1 = mth <= 6
    H2 = ~H1
    rid = df_raw.loc[df_raw['season'] == vs, 'row_id'].to_numpy()
    pos = {r: i for i, r in enumerate(rid)}

    print(f'\n=== fold {tag} (train<={vs-1} -> {vs}) : 8헤드 근사 블렌드={sc(blend, np.ones(len(yv),bool)):.2f} ===')
    for f, col in [(f'dev/phase3_preds/fold_{vs}_pred_tabm_base.csv', 'pred_tabm_base'),
                   (f'dev/phase3_preds/fold_{vs}_pred_tabm_pwl.csv', 'pred_tabm_pwl')]:
        if not os.path.exists(f):
            print(f'  {col}: 파일없음')
            continue
        d_ = pd.read_csv(f)
        idx = np.array([pos.get(r, -1) for r in d_['row_id'].to_numpy()])
        ok = idx >= 0
        p = np.full(len(blend), np.nan)
        p[idx[ok]] = d_[col].to_numpy()[ok]
        p = np.where(np.isnan(p), blend, p)
        d = p - blend
        md, mr = d.mean(), resid.mean()
        cov = np.mean((d - md) * (resid - mr))
        var = np.mean((d - md) ** 2)
        maxg = (cov * cov / var) * K
        gains, coefs = [], []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            mdf, mrf = d[fit_m].mean(), resid[fit_m].mean()
            cv = np.mean((d[fit_m] - mdf) * (resid[fit_m] - mrf))
            vr = np.mean((d[fit_m] - mdf) ** 2)
            a = cv / vr if vr > 1e-14 else 0.0
            b = mrf - a * mdf
            bl = blend.copy()
            bl[ev_m] = blend[ev_m] + a * d[ev_m] + b
            gains.append(sc(bl, ev_m) - sc(blend, ev_m))
            coefs.append(a)
        print(f'  {col:16s} 단독={sc(p, np.ones(len(yv),bool)):8.1f}  중심화최대={maxg:+7.2f}  '
              f'a={coefs[0]:+.3f}/{coefs[1]:+.3f}  H1->H2={gains[0]:+8.2f}  H2->H1={gains[1]:+8.2f}  평균={np.mean(gains):+8.2f}')

print()
print('=== 배포 가능성 점검 ===')
for mod in ['torch', 'tabm', 'rtdl_num_embeddings']:
    try:
        m = __import__(mod)
        print(f'  {mod:22s} 설치됨  ver={getattr(m, "__version__", "?")}')
    except Exception as e:
        print(f'  {mod:22s} 없음 ({type(e).__name__})')
try:
    import torch
    print(f'  CUDA 사용가능 = {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  GPU = {torch.cuda.get_device_name(0)}')
except Exception:
    pass
