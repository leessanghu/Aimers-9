"""시드 노이즈 Var(eps) 측정 -> 시드배깅으로 얻을 수 있는 점수 상한.
p = p_true + eps 이면 BS = BS_true + Var(eps).
시드 2개가 캐시된 헤드에서 d = p_s42 - p_s7 -> Var(eps) = Var(d)/2."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
yv = meta['control_success'].to_numpy(np.float64)[season == 2024]

v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
W = {k: float(v88[f'{k}_weight']) for k in
     ['base', 'hurdle', 'multires', 'ordinal', 'midother',
      'condball', 'countresid', 'future50', 'mc5', 'ingame']}

PAIRS = {
    'multires':   [f'dev/idea13_cache/A_multires_s{s}.npy' for s in (42, 7)],
    'ordinal':    [f'dev/idea13_cache/A_ordinal_s{s}.npy' for s in (42, 7)],
    'midother':   [f'dev/idea46_cache/A_midother_s{s}.npy' for s in (42, 7)],
    'condball':   [f'dev/idea54_cache/A_cond_ball_s{s}.npy' for s in (42, 7)],
    'countresid': [f'dev/idea54_cache/A_count_resid_s{s}.npy' for s in (42, 7)],
    'future50':   [f'dev/idea54_cache/A_future50_multi_s{s}.npy' for s in (42, 7)],
}

print(f'{"헤드":12s} {"가중치":>7s} {"시드간std":>10s} {"Var(eps)":>11s} {"블렌드기여Var":>13s} {"2->inf 추가이득":>15s}')
tot_var_now = 0.0
rows = []
for head, ps in PAIRS.items():
    if not all(os.path.exists(p) for p in ps):
        print(f'  {head}: 캐시없음')
        continue
    a, b = np.load(ps[0]), np.load(ps[1])
    d = a - b
    var_eps = float(d.var()) / 2.0          # 시드1개의 노이즈 분산
    w = W[head]
    # 현재 이 헤드는 시드2개 평균 -> 남은 노이즈 = var_eps/2
    remain = var_eps / 2.0
    contrib = (w ** 2) * remain              # 블렌드 예측에 남아있는 노이즈 분산
    gain_inf = contrib * K                   # 무한시드로 완전제거시 이득
    tot_var_now += contrib
    rows.append((head, w, float(d.std()), var_eps, contrib, gain_inf))
    print(f'  {head:10s} {w:7.4f} {d.std():10.6f} {var_eps:11.3e} {contrib:13.3e} {gain_inf:+15.2f}')

print()
print(f'  (시드2개 이미 적용중인 6개 헤드) 남은 노이즈 총합 = {tot_var_now:.3e}')
print(f'  -> 이 헤드들에서 시드를 무한대로 늘릴 때 최대이득 = {tot_var_now*K:+.2f}점')

print()
print('=== 단일시드 헤드(base/hurdle/mc5/ingame) 추정 ===')
if rows:
    typical = float(np.median([r[3] for r in rows]))
    print(f'  측정된 헤드들의 Var(eps) 중앙값 = {typical:.3e}  (시드간std ~{np.sqrt(2*typical):.5f})')
    single = ['base', 'hurdle', 'mc5', 'ingame']
    tot_single = 0.0
    for h in single:
        w = W[h]
        contrib = (w ** 2) * typical      # 시드1개 -> 노이즈 그대로
        tot_single += contrib
        print(f'  {h:10s} w={w:.4f}  가정 노이즈기여={contrib:.3e}  완전제거시 {contrib*K:+.2f}점')
    print(f'  -> 단일시드 4개 헤드 합계 최대이득 = {tot_single*K:+.2f}점')
    print()
    print(f'  ★ 전체 시드배깅 이론상한 = {(tot_var_now+tot_single)*K:+.2f}점')
    for k in (3, 5, 10):
        # 단일->k시드: 노이즈 1/k로. 2시드->k시드: (1/2 - 1/k)만큼 제거
        g = (tot_single * (1 - 1.0 / k) + tot_var_now * (1 - 2.0 / k) if k >= 2 else 0) * K
        print(f'    시드 {k:2d}개 사용시 예상이득 = {g:+.2f}점')
