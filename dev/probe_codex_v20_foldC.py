"""codex v20을 fold C(2022)에서 평가 + fold A와 오염배율 비교."""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
PKG = ('C:/Users/이상후/AppData/Local/Temp/claude/'
       'c--Users-----OneDrive-------Aimers-9/3064dbc5-47a2-47b1-b613-f1f60c5848ac/scratchpad/codex_v20')

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


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


def our_blend(tag, vs):
    va = season == vs
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t = sum(W.values()); W = {k: v / t for k, v in W.items()}
    return np.clip(sum(W[k] * H[k] for k in H), 0, 1), y_all[va]


rows = [dict(fold='A', ours=923.82, codex=1651.62, ratio=1651.62 / 923.82,
             cp=0.9214, cr=0.1251, s_opt=1.3287, gain=775.25)]   # 앞선 실행에서 측정됨
for tag, vs, subfile, infile in [
        ('C', 2022, f'{PKG}/output/submission_foldC.csv', 'dev/codex_foldC_input/test.csv')]:
    blend, yv = our_blend(tag, vs)
    sub = pd.read_csv(subfile, encoding='utf-8')
    order = pd.read_csv(infile, encoding='utf-8', usecols=['row_id'])
    p = order[['row_id']].merge(sub, on='row_id', how='left')['control_success'].to_numpy(np.float64)
    assert np.isfinite(p).all() and len(p) == len(yv)
    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yv) ** 2) / B)
    d = p - blend
    resid = yv - blend
    C = float(np.mean((blend - yv) * d)); V = float(np.mean(d * d))
    rows.append(dict(fold=tag, ours=sc(blend), codex=sc(p),
                     ratio=sc(p) / sc(blend),
                     cp=np.corrcoef(p, blend)[0, 1],
                     cr=np.corrcoef(d, resid)[0, 1],
                     s_opt=-C / V, gain=K * C ** 2 / V))

print(f'{"fold":<6}{"우리v95":>10}{"codex":>10}{"배율":>8}{"예측corr":>10}{"잔차corr":>10}{"s*":>9}{"최대이득":>11}')
for r in rows:
    print(f'{r["fold"]:<6}{r["ours"]:>10.1f}{r["codex"]:>10.1f}{r["ratio"]:>8.2f}'
          f'{r["cp"]:>+10.4f}{r["cr"]:>+10.4f}{r["s_opt"]:>+9.3f}{r["gain"]:>+11.1f}')

print('\ncodex 자체보고 metadata: bss=905.2  base_rawid_half=897.8  second_half=764.5')
print('우리 v95 실측(리더보드) = 1103.66')
print('\n[진단] 두 fold 모두 codex가 우리보다 훨씬 높게 나오면 = 두 fold 다 in-sample 오염.')
print('       그러면 이 로컬 수치로는 앙상블 가치를 전혀 판정할 수 없다.')
