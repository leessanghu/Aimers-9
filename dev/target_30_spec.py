"""'+30점'을 모든 의미있는 단위로 환산 — 후보 아이디어 사전판정용 스펙시트.

Score = 1e5 * (1 - BS/BSref),  BSref = 0.249807,  K = 1e5/BSref
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

BSREF = 0.249807
K = 1e5 / BSREF
CUR = 1103.6568315036
TARGET = CUR + 30.0

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


bs_cur = BSREF * (1 - CUR / 1e5)
bs_tgt = BSREF * (1 - TARGET / 1e5)
dbs = bs_cur - bs_tgt

print('=' * 78)
print(f'  현재 v95 = {CUR:.4f}   목표 = {TARGET:.4f}   (+30.00점)')
print('=' * 78)
print(f'\n[1] Brier 스케일')
print(f'  현재 BS   = {bs_cur:.8f}')
print(f'  목표 BS   = {bs_tgt:.8f}')
print(f'  필요 ΔBS  = {dbs:.8f}   ({dbs/bs_cur*100:.4f}% 감소)')

# fold A 기준 실제 잔차 통계
va = season == 2024
yv = y_all[va]
H = build8('A')
W = {k: float(v95[f'{k}_weight']) for k in H}
t = sum(W.values()); W = {k: v / t for k, v in W.items()}
blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
resid = yv - blend
var_r = float(np.mean(resid ** 2))
rbar = yv.mean()
var_label = rbar * (1 - rbar)
explained_now = var_label - float(np.mean((blend - yv) ** 2))

print(f'\n[2] 설명분산 스케일 (fold A 기준)')
print(f'  라벨 총분산            = {var_label:.6f}')
print(f'  v95가 현재 설명하는 양 = {explained_now:.6f}  (총분산의 {explained_now/var_label*100:.3f}%)')
print(f'  +30점에 필요한 추가량  = {dbs:.8f}')
print(f'  => 설명분산을 {dbs/explained_now*100:.2f}% 더 늘려야 함 (상대증가)')
print(f'  => 총 라벨분산 기준으로는 추가 {dbs/var_label*100:.4f}%p 설명')

print(f'\n[3] 신규신호 d가 만족해야 할 조건  (최대이득 = K*Cov(d,r)^2/Var(d))')
print(f'  잔차분산 Var(r) = {var_r:.6f}')
rho_need = np.sqrt(30.0 / (K * var_r))
print(f'  필요 corr(d, resid) = {rho_need:.5f}')
print(f'  필요 |Cov(d,r)|/sd(d) = {np.sqrt(30.0/K):.6f}')
print(f'\n  [오늘 측정된 후보들의 실제 잔차상관]')
cands = [('xgb_rawid', -0.0076), ('xgb_ctx', -0.0073), ('lgbm_rawid', -0.0072),
         ('xgb_hurdle_ctx', -0.0035), ('persona', -0.0058),
         ('base(우리 최고 헤드)', +0.0036)]
for nm, r in cands:
    ach = K * r ** 2 * var_r
    print(f'    {nm:<22} rho={r:+.4f}  (필요치의 {abs(r)/rho_need*100:5.1f}%)  '
          f'부호{"O" if r > 0 else "X"}  최대이득={ach:+6.2f}점')

print(f'\n[4] 앙상블 ambiguity 스케일')
amb_now = sum(W[k] * float(np.mean((np.clip(H[k], 0, 1) - blend) ** 2)) for k in H)
print(f'  현재 ambiguity = {amb_now:.6f}  (= +{K*amb_now:.1f}점을 벌고 있음)')
print(f'  +30점을 ambiguity만으로 = {amb_now + 30/K:.6f}  (현재의 {(amb_now+30/K)/amb_now:.2f}배)')
print(f'  => 8헤드가 +{K*amb_now:.1f}점 버는데, 그걸 {(amb_now+30/K)/amb_now:.2f}배로 = 헤드 다양성 {((amb_now+30/K)/amb_now-1)*100:.0f}% 증가 필요')
print(f'  => 지금 헤드와 동급 품질의 새 타겟분해 헤드 약 {8*((amb_now+30/K)/amb_now-1):.0f}개 상당')

print(f'\n[5] 측정 관점 (오늘 실측한 SE)')
print(f'  가중치 미세조정 SE ≈ 0.14~1.06점  -> +30은 z=28~214 (압도적으로 검출가능)')
print(f'  완전 다른 모델   SE ≈ 12.3점      -> +30은 z=2.4 (검출가능하나 1회론 애매)')
print(f'  즉 +30짜리 효과가 진짜 있다면 제출 1회로 확실히 보인다. 못 보고 있다는 건')
print(f'  그런 크기의 효과를 아직 못 만들었다는 뜻.')

print(f'\n[6] 오늘까지 측정된 모든 축의 실측 상한 합')
axes = [('시드배깅(K=1->무한)', 1.48), ('레벨축 잔여', 0.38), ('가중치축', 0.24),
        ('risk_alpha축', 1.04), ('XGB/LGBM 전역', 0.34), ('XGB 구간별', 0.41),
        ('스태킹', 0.0), ('강제 다양화', 0.0)]
tot = sum(a[1] for a in axes)
for nm, v in axes:
    print(f'    {nm:<24} {v:+6.2f}')
print(f'    {"합계":<24} {tot:+6.2f}   <- 필요한 +30.00의 {tot/30*100:.1f}%')
