"""P1(season 외삽 실패 우려) 검증: fold A/C 자체가 이미 'season 외삽' 시나리오다.
fold A: train<=2023(season 최대값=2023) -> predict season=2024 (훈련범위 밖 최초)
fold C: train<=2021(season 최대값=2021) -> predict season=2022 (훈련범위 밖 최초)
GBDT가 season을 외삽 못 해서 큰 편향이 생긴다면, 이 두 fold의 레벨편향(E[pred]-E[actual])이
D_true(-0.00097, 2025 실측)와 다른 크기로 나타나야 한다(더 커야 함, 훈련범위 경계라서).
같은 크기로 작으면 -> '외삽 실패'가 구조적으로 크지 않다는 독립적 증거 2개 추가 확보.

추가로 season 추세(연도별 성공률)와 각 fold의 실제 편향 방향을 나란히 보여
'우연히 평평해서 작았다'는 반론에 대한 정보도 제공한다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B

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


print('=== 연도별 실제 성공률 추세 (전체 train) ===')
for yr in sorted(set(season.tolist())):
    r = y_all[season == yr].mean()
    print(f'  {int(yr)}: {r:.4f}')

print('\n=== fold별 레벨편향(E[pred]-E[actual]): season 외삽 경계에서 큰가? ===')
print(f'{"fold":<8}{"train상한":>10}{"predict":>10}{"실제성공률":>12}{"블렌드평균":>12}{"편향D":>10}{"이득손실":>10}')
for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    va = season == vs
    yv = y_all[va]
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t = sum(W.values()); W = {k: v / t for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    D = float(blend.mean() - yv.mean())
    print(f'{tag:<8}{upto:>10}{vs:>10}{yv.mean():>12.4f}{blend.mean():>12.4f}{D:>+10.5f}{K*D**2:>10.2f}')

print(f'\n실측(2025, 진짜 외삽 경계) D_true = -0.00097  손실={K*0.00097**2:.2f}점')
print('\n[해석] 세 경계(22,24,25년 첫해) 모두 |D|가 비슷하게 작으면(0.001~0.01 수준)')
print(' "외삽 실패로 인한 큰 레벨편향"은 구조적으로 재현되지 않는다는 독립 증거 3개.')
print(' 어느 한 fold라도 |D|가 확연히 크면(0.02+) P1이 부분적으로 맞고 추가조사 필요.')
