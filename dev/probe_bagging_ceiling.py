"""'시드배깅은 Var를 1/K배로 줄인다, +2는 하한'이라는 주장을 실측 검증.

올바른 분해:  Var_total = Var_D(E_seed[f])  +  E_D[Var_seed(f)]
                          ^데이터유래(배깅으로 안 줄어듦)   ^시드유래(1/K배)
따라서 배깅의 이론상한은 '현재 남아있는 시드분산'이며, 그건 직접 측정 가능하다.

우리 헤드 6개는 이미 시드 2개(s42, s7) 평균이다. p1,p2가 iid이면
  E[(p1-p2)^2] = 2*sigma^2  ->  sigma^2 = mean((p1-p2)^2)/2
현재 K=2이므로 블렌드에 남은 시드분산 = sum_k w_k^2 * sigma_k^2 / 2
K->무한대로 가면 이만큼이 BS에서 사라진다(그게 배깅 이론상한, 하한 아님).
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
KC = 1e5 / B

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')

SEED_PAIRS = {
    'multires':   'dev/idea13_cache/{t}_multires_s{s}.npy',
    'ordinal':    'dev/idea13_cache/{t}_ordinal_s{s}.npy',
    'midother':   'dev/idea46_cache/{t}_midother_s{s}.npy',
    'condball':   'dev/idea54_cache/{t}_cond_ball_s{s}.npy',
    'countresid': 'dev/idea54_cache/{t}_count_resid_s{s}.npy',
    'future50':   'dev/idea54_cache/{t}_future50_multi_s{s}.npy',
}
ALL_HEADS = ['base', 'hurdle', 'multires', 'ordinal', 'midother',
             'condball', 'countresid', 'future50']
W = {k: float(v95[f'{k}_weight']) for k in ALL_HEADS}
tot = sum(W.values()); W = {k: v / tot for k, v in W.items()}

for tag, vs in [('A', 2024), ('C', 2022)]:
    print(f'\n=== fold {tag} ({vs}) ===')
    print(f'{"헤드":<14}{"w":>8}{"sd(s42-s7)":>13}{"sigma(1시드)":>14}'
          f'{"w^2*sig^2/2":>14}{"->점수":>10}')
    total_var = 0.0
    for head, pat in SEED_PAIRS.items():
        p1 = np.load(pat.format(t=tag, s=42))
        p2 = np.load(pat.format(t=tag, s=7))
        diff = p1 - p2
        sigma2 = float(np.mean(diff ** 2)) / 2.0        # 시드 1개짜리 분산
        contrib = W[head] ** 2 * sigma2 / 2.0            # 현재 K=2라 /2
        total_var += contrib
        print(f'{head:<14}{W[head]:>8.4f}{np.std(diff):>13.5f}'
              f'{np.sqrt(sigma2):>14.5f}{contrib:>14.3e}{KC*contrib:>10.2f}')
    print(f'{"합계(시드쌍 보유 6헤드)":<14}{"":<8}{"":<13}{"":<14}'
          f'{total_var:>14.3e}{KC*total_var:>10.2f}')
    print(f'  -> K=2에서 K=무한대로 갈 때 이론 최대이득 = {KC*total_var:+.2f}점')
    print(f'  -> base/hurdle은 시드쌍 캐시가 없어 미포함(하이퍼파라미터 변종이라 순수시드 아님)')

print('\n[결론] 이 값이 배깅의 이론 "상한"이다(하한 아님).')
print(' 데이터유래 분산 Var_D(E_seed[f])는 시드를 아무리 늘려도 안 줄어들기 때문.')
