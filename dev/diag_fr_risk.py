"""F/R 리그 비중이 test에서 달라지면 위험한가?
fold A(2024)에서 F/R별로 (1) 모델 정확도(BSS) (2) 레벨편향 을 따로 재본다.
F가 최근 변동성이 커서(2022:+0.71 -> 2023:-0.24) 이 그룹만 특히 편향이 클 수 있다."""
import numpy as np, pandas as pd, joblib, sys
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
gt = meta['game_type'].to_numpy()
va = season == 2024
yv = y[va]; gtv = gt[va]; unc = 0.249807

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
HARM = ['multires', 'midother', 'condball', 'countresid', 'future50', 'ingame']
W94 = {k: (v * 0.2 if k in HARM else v) for k, v in W.items()}
t = sum(W94.values()); W94 = {k: v / t for k, v in W94.items()}
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
p94 = sum(W94[k] * H[k] for k in W94)


def sc(m):
    return 1e5 * (1 - np.mean((np.clip(p94[m], 0, 1) - yv[m]) ** 2) / unc)


F = gtv == 'F'; R = gtv == 'R'
print('=== F/R별 정확도·편향 (fold A, v94) ===')
for nm, m in [('전체', np.ones(len(yv), bool)), ('R', R), ('F', F)]:
    print(f'  {nm:4s} n={m.sum():>7,}  BSS={sc(m):8.1f}  실제={yv[m].mean():.4f}  예측={p94[m].mean():.4f}  편차={p94[m].mean()-yv[m].mean():+.4f}')
print()

print('=== 만약 test의 F 비중이 바뀐다면 전체 BSS가 어떻게 되나 (시뮬레이션) ===')
bs_R = np.mean((np.clip(p94[R], 0, 1) - yv[R]) ** 2)
bs_F = np.mean((np.clip(p94[F], 0, 1) - yv[F]) ** 2)
print(f'  R 전용 Brier={bs_R:.5f}  F 전용 Brier={bs_F:.5f}')
for f_share in (0.0, 0.05, 0.10, 0.1184, 0.15, 0.20, 0.30):
    brier_mix = (1 - f_share) * bs_R + f_share * bs_F
    r_mix = (1 - f_share) * yv[R].mean() + f_share * yv[F].mean()
    base_dyn = r_mix * (1 - r_mix)
    bss_own = 1e5 * (1 - brier_mix / base_dyn)
    bss_fixed = 1e5 * (1 - brier_mix / unc)
    print(f'  F비중={f_share:.2%}  BSS(자체baseline)={bss_own:7.1f}  BSS(고정0.2498)={bss_fixed:7.1f}')
print()
print('=== 연도별 F share 재확인 ===')
gtr = meta['game_type'].to_numpy()
for s in range(2019, 2025):
    m = season == s
    fshare = (gtr[m] == 'F').mean()
    print(f'  {s}: F비중={fshare:.4f}')
