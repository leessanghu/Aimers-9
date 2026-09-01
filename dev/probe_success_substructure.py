"""y=1의 내부구조 분해 + '포수 의도(존 안/밖)' 부분복원.

핵심 통찰:
  success=1 & ball=1   -> 포수 타겟이 존 '밖'이었다 (웨이스트/유인구/고의볼)
  success=1 & strike=1 -> 포수 타겟이 존 '안'이었다
  success=1 & 인플레이   -> 타자가 쳤음 (타겟은 대체로 존 안 근처)
  success=0            -> 타겟을 못 맞췄으므로 의도 불명 (NaN)

즉 '포수 의도'는 부분관측 잠재변수다. MultiRMSEWithMissingValues가 정확히
이런 결측 타겟을 다루므로 multi-task 보조타겟으로 쓸 수 있다.

이 스크립트는 그 구조가 진짜 신호인지 확인한다:
 (1) y=1 하위유형 분해와 상황별 변화
 (2) 존의도가 카운트/구종에 따라 체계적으로 변하는가 (= 예측가능한가)
 (3) 존의도별 제구성공률이 다른가 (= y와 인과적으로 연결되는가)
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
y = df['control_success'].to_numpy(np.float64)
pid = df['pitcher_id'].to_numpy()
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)
order = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[order]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    lab = np.empty(n); lab[order] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball = diff_label('asof_pitcher_ball_rate')
strike = diff_label('asof_pitcher_strike_rate')
ptype = np.load('dev/recovered_pitch_type.npy')
ok = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball) & np.isfinite(strike)
inplay = np.where(ok, ((ball == 0) & (strike == 0)).astype(np.float64), np.nan)

balls_b = df['balls_before'].to_numpy()
strikes_b = df['strikes_before'].to_numpy()

print('=' * 80)
print('[1] 판정축 검증: ball + strike + 인플레이 = 1 인가?')
print('=' * 80)
s = ball[ok] + strike[ok] + inplay[ok]
print(f'  합=1인 비율 {np.mean(s == 1)*100:.2f}%   ball {np.mean(ball[ok])*100:.2f}%'
      f'  strike {np.mean(strike[ok])*100:.2f}%  인플레이 {np.mean(inplay[ok])*100:.2f}%')

print('\n' + '=' * 80)
print('[2] y=1 (제구성공) 내부 분해')
print('=' * 80)
succ = ok & (y == 1)
tot_s = succ.sum()
for nm, m in [('존안(strike)', succ & (strike == 1)),
              ('존밖(ball)  ', succ & (ball == 1)),
              ('인플레이     ', succ & (inplay == 1))]:
    print(f'  {nm}  n={m.sum():>9,}  전체의 {m.sum()/n*100:5.2f}%  성공중 {m.sum()/tot_s*100:5.2f}%')

print('\n' + '=' * 80)
print('[3] y=0 (제구실패) 내부 분해')
print('=' * 80)
fail = ok & (y == 0)
tot_f = fail.sum()
for nm, m in [('reverse만  ', fail & (rev == 1) & (mid == 0)),
              ('middle만   ', fail & (rev == 0) & (mid == 1)),
              ('둘다       ', fail & (rev == 1) & (mid == 1)),
              ('기타(둘다0)', fail & (rev == 0) & (mid == 0))]:
    print(f'  {nm}  n={m.sum():>9,}  전체의 {m.sum()/n*100:5.2f}%  실패중 {m.sum()/tot_f*100:5.2f}%')

print('\n' + '=' * 80)
print('[4] 포수 존의도 복원 — 카운트별 (성공행에서만 관측 가능)')
print('=' * 80)
# zone_intent: 성공행에서만 정의. 1=존안 요구(strike or 인플레이), 0=존밖 요구(ball)
zone_intent = np.where(succ, 1.0 - ball, np.nan)
print(f'  복원 가능 행: {succ.sum():,} ({succ.sum()/n*100:.1f}%)  나머지는 결측(NaN)')
print(f'\n{"카운트":<10}{"n(성공)":>10}{"존안요구율":>12}{"전체성공률":>12}{"직구비율":>10}')
for b in range(4):
    for s_ in range(3):
        m = ok & (balls_b == b) & (strikes_b == s_)
        ms = m & succ
        if ms.sum() < 300:
            continue
        fb = np.mean(ptype[m] == 0) if (ptype[m] >= 0).any() else np.nan
        print(f'  {b}-{s_:<7}{ms.sum():>10,}{np.nanmean(zone_intent[ms])*100:>11.1f}%'
              f'{y[m].mean()*100:>11.1f}%{fb*100:>9.1f}%')

print('\n' + '=' * 80)
print('[5] 존의도 x 구종 교차 (구종이 존의도를 예측하는가)')
print('=' * 80)
for t, nm in [(0, '직구'), (1, '변화구'), (2, '오프스피드')]:
    m = succ & (ptype == t)
    if m.sum() < 500:
        continue
    print(f'  {nm:<8} n={m.sum():>9,}  존안요구율 {np.nanmean(zone_intent[m])*100:5.1f}%')

print('\n' + '=' * 80)
print('[6] 핵심질문: 존밖 요구 상황에서 제구성공률이 다른가?')
print('=' * 80)
print('  (존의도는 성공행에서만 관측되므로 직접비교 불가.')
print('   대신 "존밖요구가 많은 카운트"에서 전체 성공률이 어떤지 본다)')
rows = []
for b in range(4):
    for s_ in range(3):
        m = ok & (balls_b == b) & (strikes_b == s_)
        ms = m & succ
        if ms.sum() < 300:
            continue
        rows.append((f'{b}-{s_}', np.nanmean(zone_intent[ms]), y[m].mean(), m.sum()))
arr = np.array([(r[1], r[2]) for r in rows])
cc = np.corrcoef(arr[:, 0], arr[:, 1])[0, 1]
print(f'\n  카운트 12칸 수준에서 corr(존안요구율, 전체성공률) = {cc:+.4f}')
print(f'  -> 음수면 "존밖 요구가 많은 카운트일수록 제구성공률 높음"')
print(f'     (존밖 타겟이 더 맞추기 쉽다 = 여유있는 카운트)')

np.save('dev/recovered_zone_intent.npy', zone_intent)
np.save('dev/recovered_call_axis.npy', np.column_stack([ball, strike, inplay]))
print(f'\n저장: dev/recovered_zone_intent.npy (성공행만 유효, 나머지 NaN)')
print(f'      dev/recovered_call_axis.npy [ball, strike, inplay]')
