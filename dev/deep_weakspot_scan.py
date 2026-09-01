"""캐시만 사용(재학습 없음). 두 가지 정밀검증:
(1) game_type(R/F) 판별력 direct 비교, fold A/C
(2) 상위 연속형 피처 10개 십분위 reliability curve, |z|>3 유의어긋남 + fold A/C 재현"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
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


data = {}
for tag, vs in [('A', 2024), ('C', 2022)]:
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v88[f'{k}_weight']) for k in H}
    t = sum(W.values())
    W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    raw = raw_all[raw_all['season'] == vs].reset_index(drop=True)
    Xv = X.loc[va].reset_index(drop=True)
    data[tag] = (yv, blend, raw, Xv)

print('=== (1) game_type(R/F) 판별력 direct 비교 ===')
for tag in ('A', 'C'):
    yv, blend, raw, Xv = data[tag]
    for gt in ('F', 'R'):
        m = (raw['game_type'] == gt).to_numpy()
        yy, pp = yv[m], blend[m]
        r = yy.mean(); var_own = max(r * (1 - r), 1e-6)
        bs = np.mean((pp - yy) ** 2)
        corr = np.corrcoef(pp, yy)[0, 1]
        print(f'  fold{tag} {gt}: n={m.sum():>7,}  corr={corr:+.4f}  자체BSS={1e5*(1-bs/var_own):7.1f}  편차={pp.mean()-r:+.5f}')

print('\n=== (2) 연속형 주요피처 십분위 reliability curve (fold A) ===')
yv, blend, raw, Xv = data['A']
resid = yv - blend
TOPFEATS = ['x_ability_here', 'inseason_cmd_index', 'bat_inseason_smooth', 'season',
            'x_count_pressure', 'asof_pitcher_offspeed_rate_smooth', 'strikes_before',
            'inseason_success_smooth', 'same_hand', 'batter_team_id_te']

flagged_all = {}
for feat in TOPFEATS:
    if feat not in Xv.columns:
        continue
    vals = Xv[feat].to_numpy(np.float64)
    order = np.argsort(vals)
    K = 20
    edges = np.linspace(0, len(vals), K + 1).astype(int)
    flagged = []
    for i in range(K):
        s, e = edges[i], edges[i+1]
        idx = order[s:e]
        m_act = yv[idx].mean(); m_pred = blend[idx].mean(); n = e - s
        se = np.sqrt(max(m_act * (1 - m_act), 1e-9) / n)
        z = (m_act - m_pred) / se if se > 0 else 0
        if abs(z) > 3:
            flagged.append((i, vals[idx].mean(), m_pred, m_act, z))
    flagged_all[feat] = flagged
    tag = f'  {feat:32s} |z|>3 구간 = {len(flagged)}/20'
    if flagged:
        tag += '  ' + ', '.join(f'[{i}]z={z:+.1f}' for i, _, _, _, z in flagged)
    print(tag)

print('\n=== fold C 재현 확인 (fold A에서 어긋난 피처만) ===')
yv_c, blend_c, raw_c, Xv_c = data['C']
for feat, flags in flagged_all.items():
    if not flags:
        continue
    vals_c = Xv_c[feat].to_numpy(np.float64) if feat in Xv_c.columns else None
    if vals_c is None:
        continue
    order_c = np.argsort(vals_c)
    K = 20
    edges_c = np.linspace(0, len(vals_c), K + 1).astype(int)
    print(f'  [{feat}] fold A 어긋난 구간의 fold C 값:')
    for i, vmean_a, mp_a, ma_a, z_a in flags:
        s, e = edges_c[i], edges_c[i+1]
        idx = order_c[s:e]
        m_act_c = yv_c[idx].mean(); m_pred_c = blend_c[idx].mean(); n = e - s
        se_c = np.sqrt(max(m_act_c * (1 - m_act_c), 1e-9) / n)
        z_c = (m_act_c - m_pred_c) / se_c if se_c > 0 else 0
        same_sign = 'O' if np.sign(z_a) == np.sign(z_c) else 'X'
        print(f'    구간{i}: foldA z={z_a:+.1f}  foldC z={z_c:+.1f}  부호일치={same_sign}')
