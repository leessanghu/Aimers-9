"""idea88b(팀원 정직재학습) + v88_final을 캐시에서 불러와 2024 검증 기준
최적 블렌드 가중치를 그리드서치 + H1/H2 양방향 안정성 확인.
idea88b/idea89가 만든 npy 캐시를 사용 (없으면 스킵 안내)."""
import numpy as np, pandas as pd, sys, os
sys.stdout.reconfigure(encoding='utf-8')

X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy(); y = meta['control_success'].to_numpy(np.float64)
va = season == 2024
yv = y[va]
mth = X.loc[va, 'game_month'].to_numpy()
unc = 0.249807
sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / unc)

v88_final = np.load('dev/cache_v88_final_2024.npy')
teammate = np.load('dev/cache_teammate_honest_2024.npy')

assert len(v88_final) == len(yv) == len(teammate), (len(v88_final), len(yv), len(teammate))

H1 = mth <= 6; H2 = ~H1
allm = np.ones(len(yv), bool)

print(f'v88_final 단독 = {sc(v88_final, allm):.1f}')
print(f'teammate 단독  = {sc(teammate, allm):.1f}')
print(f'corr(pred) = {np.corrcoef(v88_final, teammate)[0,1]:.4f}')
err_v = v88_final - yv; err_t = teammate - yv
print(f'corr(err)  = {np.corrcoef(err_v, err_t)[0,1]:.4f}')
print()

print('=== 그리드서치 (전체 2024) ===')
best_w, best_s = 0.0, sc(v88_final, allm)
for w in np.arange(0.0, 0.51, 0.02):
    blend = (1 - w) * v88_final + w * teammate
    s = sc(blend, allm)
    if s > best_s:
        best_s, best_w = s, w
    if round(w * 50) % 5 == 0:
        print(f'  w={w:.2f}  BSS={s:.2f}  ({s-sc(v88_final,allm):+.2f})')
print(f'>>> 최적 w={best_w:.3f}  BSS={best_s:.2f}  (v88단독 대비 {best_s-sc(v88_final,allm):+.2f})')
print()

print('=== H1<->H2 정직 검증 (fit구간에서 C/V로 w* 추정 -> eval구간 적용) ===')
resid_v = yv - v88_final
d = teammate - v88_final
gains = []
for fit_m, ev_m, tag in [(H1, H2, 'H1->H2'), (H2, H1, 'H2->H1')]:
    C = np.mean(d[fit_m] * resid_v[fit_m]); V = np.mean(d[fit_m] ** 2)
    w_star = C / V if V > 1e-12 else 0.0
    blend = v88_final.copy(); blend[ev_m] = v88_final[ev_m] + w_star * d[ev_m]
    g = sc(blend, ev_m) - sc(v88_final, ev_m)
    gains.append(g)
    print(f'  {tag}: fit에서 구한 w*={w_star:.4f}  eval구간 이득={g:+.2f}')
print(f'평균 이득 = {np.mean(gains):+.2f}')
