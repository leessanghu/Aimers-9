"""복원된 구종(직구/변화구/오프스피드)이 새 정보채널인지 진단.

Rule4 안전성:
 - 구종은 '현재 투구'의 것이라 test에서는 알 수 없다(다음 행 참조 필요). 따라서
   피처로 직접 못 쓴다. 대신 두 가지 용법이 가능:
   (a) 학습데이터로 만든 (투수, 구종) 룩업테이블 -> test에서 pitcher_id로 조회 (안전)
   (b) multi-task 보조타겟 y (추론시 head0만 사용) -> 기존 헤드들과 동일한 방식 (안전)

이 스크립트는 (a) 쪽 신호를 먼저 싸게 측정한다:
   구종별 제구성공률의 '투수간 편차'가 잔차와 상관이 있는가?
   +30점 스펙: corr(d, resid) = 0.0174 필요.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B = 0.249807
K = 1e5 / B
NEED_RHO = 0.01740

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
n = len(df)
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
pid = df['pitcher_id'].to_numpy()
ptype = np.load('dev/recovered_pitch_type.npy')   # 0=직구 1=변화구 2=오프스피드, -1=복원불가
cs = (df['balls_before'].to_numpy() * 4 + df['strikes_before'].to_numpy())

meta = pd.read_parquet('dev/featcache_meta.parquet')
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{m}.npy' for m in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{m}.npy')) *
                        np.load(f'dev/phase90_cache/{tag}_snc_{m}.npy') for m in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


print('=== 구종 신호의 크기 진단 ===')
ok = ptype >= 0
print(f'복원 커버리지 {ok.mean()*100:.1f}%')
print(f'\n[A] 구종별 전역 성공률')
for t, nm in [(0, '직구'), (1, '변화구'), (2, '오프스피드')]:
    m = ok & (ptype == t)
    print(f'  {nm:<8} n={m.sum():>9,}  success={y[m].mean():.4f}')
gap = y[ok & (ptype == 0)].mean() - y[ok & (ptype == 1)].mean()
print(f'  직구-변화구 격차 = {gap:.4f} ({gap*100:.2f}%p)')

print(f'\n[B] 구종x카운트 교차 (구종이 카운트에 따라 달라지는가 = 예측가능한가)')
for t, nm in [(0, '직구'), (1, '변화구'), (2, '오프스피드')]:
    rates = []
    for c in [0, 3, 12, 15]:   # 0-0, 0-3(0-3?) 실제는 balls*4+strikes
        m = ok & (cs == c)
        if m.sum() > 500:
            rates.append(f'{(ptype[m]==t).mean()*100:.0f}%')
        else:
            rates.append('-')
    print(f'  {nm:<8} 카운트별 사용률(cs=0,3,12,15): {"  ".join(rates)}')

print(f'\n[C] 투수별 "구종의존도" — 이게 신규 신호의 핵심')
print('    (구종별 성공률 편차가 큰 투수 = 구종에 따라 제구가 크게 달라지는 투수)')
for tag, upto, vs in [('A', 2023, 2024), ('C', 2021, 2022)]:
    tr = (season <= upto) & ok
    va = season == vs
    yv = y[va]
    H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend

    # (투수, 구종) 성공률 축소테이블 : train 파티션에서만
    g = float(y[tr].mean())
    tab = pd.DataFrame({'p': pid[tr], 't': ptype[tr], 'y': y[tr]})
    pt_agg = tab.groupby(['p', 't'])['y'].agg(['sum', 'count'])
    p_agg = tab.groupby('p')['y'].agg(['sum', 'count'])
    K_SH = 80.0
    p_rate = ((p_agg['sum'] + 100 * g) / (p_agg['count'] + 100)).rename('prate')
    pt_j = pt_agg.join(p_rate, on='p')
    pt_j['rate'] = (pt_j['sum'] + K_SH * pt_j['prate']) / (pt_j['count'] + K_SH)
    # 투수별: 구종간 성공률 spread + 구종믹스 가중 기대값
    mix = tab.groupby(['p', 't']).size().rename('mn').reset_index()
    mix_tot = mix.groupby('p')['mn'].sum().rename('mtot')
    mix = mix.join(mix_tot, on='p')
    mix['w'] = mix['mn'] / mix['mtot']
    mm = mix.set_index(['p', 't'])[['w']].join(pt_j[['rate']])
    spread = mm.groupby('p').apply(
        lambda s: float(np.sqrt(np.average((s['rate'] - np.average(s['rate'], weights=s['w'])) ** 2,
                                           weights=s['w']))), include_groups=False).rename('spread')
    # 검증구간에 조회
    va_pid = pid[va]
    f_spread = pd.Series(va_pid).map(spread).fillna(spread.median()).to_numpy(np.float64)

    def maxgain(d):
        d = d - d.mean()
        V = float(np.mean(d ** 2))
        if V < 1e-14:
            return 0.0, 0.0
        C = float(np.mean(d * resid))
        return K * C ** 2 / V, C / np.sqrt(V * float(np.mean(resid ** 2)))

    mg, rho = maxgain(f_spread)
    print(f'\n  fold {tag}: 투수 구종의존도(spread) 피처')
    print(f'    spread 분포: 중앙값={np.median(f_spread):.4f}  '
          f'p10={np.percentile(f_spread,10):.4f}  p90={np.percentile(f_spread,90):.4f}')
    print(f'    잔차상관 rho = {rho:+.5f}   (+30점 필요치 {NEED_RHO:.5f}의 {abs(rho)/NEED_RHO*100:.1f}%)')
    print(f'    최대이득 = {mg:+.2f}점')
