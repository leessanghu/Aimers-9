"""risk_alpha 최적값을 '이미 있는 실측'으로 풀 수 있는지 확인.
(0) v87 vs v88 아티팩트가 risk 파라미터만 다른지 검증 -> 그래야 +6.23이 깨끗한 A/B
(1) E[g^2] (g = -(cut - center)) 를 fold A로 추정
(2) 알려진 실측이득으로 E[g*resid] 역산 -> 최적 alpha, 남은 이득"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, joblib

B = 0.249807
K = 1e5 / B

v87 = joblib.load('submit/model/model_artifacts_v87.pkl')
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')

print('=== (0) v87 vs v88 차이 (risk 파라미터만 다른가?) ===')
keys = sorted(set(v87) | set(v88))
ndiff = 0
for k in keys:
    if k not in v87:
        print(f'  [{k}] v87에 없음 -> v88={v88[k] if np.isscalar(v88[k]) else type(v88[k]).__name__}')
        ndiff += 1
        continue
    if k not in v88:
        print(f'  [{k}] v88에 없음')
        ndiff += 1
        continue
    a, b = v87[k], v88[k]
    if a is b:
        continue
    if isinstance(a, (int, float, str, bool, type(None))) and isinstance(b, (int, float, str, bool, type(None))):
        if a != b:
            print(f'  [{k}]  v87={a!r}   v88={b!r}')
            ndiff += 1
    elif isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
        if a.shape != b.shape or not np.array_equal(a, b):
            print(f'  [{k}]  ndarray 다름 shape {a.shape} vs {b.shape}')
            ndiff += 1
print(f'  -> 스칼라/배열 차이 개수 = {ndiff}  (모델객체는 참조가 달라도 여기선 미판정)')

print()
print('=== (1) g = -(cut - center) 의 E[g^2] (fold A 기준) ===')
P11 = np.load('dev/idea75_cache/A_proba11.npy')
risk = P11[:, [9, 10]].sum(axis=1)
thr = float(v88['risk_thr'])
center = float(v88['risk_center'])
alpha = float(v88['risk_alpha'])
cut = np.maximum(0.0, risk - thr)
g = -(cut - center)
Eg2 = float(np.mean(g ** 2))
print(f'  risk_thr={thr}  risk_center={center}  risk_alpha={alpha}')
print(f'  E[cut]={cut.mean():.6f}   (center와 비교: {center:.6f})')
print(f'  Var(cut)={cut.var():.6f}   E[g^2]={Eg2:.6f}')
print(f'  적용행 비율 = {(cut>0).mean()*100:.1f}%')

print()
print('=== (2) 알려진 실측이득으로 E[g*resid] 역산 ===')
print('  gain(a) = [2a*E[g*r] - a^2*E[g^2]] * K   (a=0에서 a로 갈 때)')
for gain_pts in (6.23,):
    # 2a*C - a^2*Eg2 = gain/K
    rhs = gain_pts / K
    C = (rhs + alpha * alpha * Eg2) / (2 * alpha)
    a_star = C / Eg2
    max_gain = (C * C / Eg2) * K
    remain = ((a_star - alpha) ** 2) * Eg2 * K
    print(f'  실측이득 {gain_pts:+.2f}점 가정:')
    print(f'    E[g*resid] = {C:.8f}')
    print(f'    최적 alpha a* = {a_star:.5f}   (현재 {alpha})')
    print(f'    a=0 -> a* 전체이득 = {max_gain:+.2f}점')
    print(f'    현재({alpha}) -> a* 남은이득 = {remain:+.2f}점')

print()
print('=== (3) 참고: fold A에서 직접 잰 최적 alpha ===')
meta = __import__('pandas').read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
yv = meta['control_success'].to_numpy(np.float64)[season == 2024]
base = np.load('dev/cache_v88_final_2024.npy')
# base는 이미 alpha=0.045가 적용된 상태 -> 되돌린 뒤 최적 재추정
p_noadj = base - alpha * g
r = yv - p_noadj
C_local = float(np.mean(g * r))
a_local = C_local / Eg2
print(f'  fold A: E[g*r]={C_local:.8f}  ->  최적 alpha = {a_local:.5f}')
print(f'  fold A 기준 현재->최적 이득 = {((a_local-alpha)**2)*Eg2*K:+.2f}점')
