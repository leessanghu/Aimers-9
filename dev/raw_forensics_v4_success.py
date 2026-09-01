"""raw 포렌식 4차 — '제구 성공'을 더 잘게 구체화하는 축 탐색.

관측된 패턴: 성공의 하위유형 분해(mc6) = +9.77점 vs 시간축(strk) = +1.10점
=> '성공이 어떤 종류의 성공인가'를 구체화하는 게 압도적으로 강하다.

mc6가 쓴 분해: 성공 = {존안(strike), 존밖(ball), 인플레이} 3분할
아직 안 쓴 추가 구체화 축을 찾는다:
 (1) 성공 x 카운트상태 : 어떤 카운트에서의 성공인가 (유리/불리/풀카운트)
 (2) 성공 x 주자상황  : 압박 상황에서의 성공인가
 (3) 성공 x 이닝후반  : 피로 상태에서의 성공인가
 (4) 성공 x 구종      : 어떤 구종으로 성공했나 (pitchtype과 결합)
 (5) 성공 x 타자손    : 좌/우 타자 상대 성공
 (6) 실패쪽 추가분할  : wild(크게벗어남)를 ball/strike/inplay로 재분할
각 조합의 성공률 편차와 표본크기를 보고 '순수 클래스'가 되는지 확인.
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
o = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[o[:-1]] = (pid[o][1:] == pid[o][:-1])


def diff_label(col):
    c = np.round(df[col].fillna(0).to_numpy(np.float64) * n_)
    c_ord = c[o]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[o]] = np.nan
    lab = np.empty(n); lab[o] = d
    return lab


rev = diff_label('asof_pitcher_reverse_rate')
mid = diff_label('asof_pitcher_middle_rate')
ball = diff_label('asof_pitcher_ball_rate')
strike = diff_label('asof_pitcher_strike_rate')
ptype = np.load('dev/recovered_pitch_type.npy')
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball) & np.isfinite(strike)
inplay = ((ball == 0) & (strike == 0)).astype(np.float64)
nd = valid & (mid < 0.5) & (rev < 0.5)

balls = df['balls_before'].to_numpy()
strikes = df['strikes_before'].to_numpy()
inning = df['inning'].to_numpy()
runners = df['num_runners_on'].to_numpy()
r2 = df['runner_on_2b'].to_numpy(); r3 = df['runner_on_3b'].to_numpy()
bhand = df['batter_hand'].to_numpy(); phand = df['pitcher_hand'].to_numpy()
g = y.mean()
print(f'전역 성공률 = {g:.4f}   유효행 = {valid.sum():,}\n')

# mc6 기존 6분할 재확인
succ = valid & (y == 1)
print('=' * 82)
print('[기준] mc6가 이미 쓰는 6분할')
print('=' * 82)
base_cls = {
    'middle':    valid & (mid > 0.5),
    'reverse':   valid & (rev > 0.5) & (mid < 0.5),
    'wild':      nd & (y == 0),
    'succ_ball': nd & (y == 1) & (ball > 0.5),
    'succ_strk': nd & (y == 1) & (strike > 0.5),
    'succ_play': nd & (y == 1) & (inplay > 0.5),
}
for nm, m in base_cls.items():
    print(f'  {nm:<12} n={m.sum():>9,} ({m.mean()*100:5.2f}%)  성공률={y[m].mean()*100:6.2f}%')

print('\n' + '=' * 82)
print('[신규후보 1] 성공 3유형 x 카운트유리도 (9분할)')
print('=' * 82)
cnt_adv = np.where(strikes > balls, 0, np.where(balls > strikes, 2, 1))   # 0=투수유리 1=중립 2=타자유리
cnames = {0: '투수유리', 1: '중립', 2: '타자유리'}
snames = {'ball': nd & (y == 1) & (ball > 0.5),
          'strk': nd & (y == 1) & (strike > 0.5),
          'play': nd & (y == 1) & (inplay > 0.5)}
print(f'{"성공유형":<8}{"카운트":<10}{"n":>10}{"전체비중":>10}')
for sn, sm in snames.items():
    for c in range(3):
        m = sm & (cnt_adv == c)
        print(f'  {sn:<8}{cnames[c]:<10}{m.sum():>10,}{m.mean()*100:>9.2f}%')

print('\n' + '=' * 82)
print('[신규후보 2] wild(크게벗어남)를 판정축으로 재분할')
print('=' * 82)
wild = nd & (y == 0)
for nm, m in [('wild_ball', wild & (ball > 0.5)),
              ('wild_strk', wild & (strike > 0.5)),
              ('wild_play', wild & (inplay > 0.5))]:
    print(f'  {nm:<12} n={m.sum():>9,} ({m.mean()*100:5.2f}%)  '
          f'wild중 {m.sum()/wild.sum()*100:5.2f}%')
print('  -> wild_strk = 크게벗어났는데 타자가 헛스윙/파울 (타자 유인 성공)')
print('     wild_play = 크게벗어났는데 타자가 침')

print('\n' + '=' * 82)
print('[신규후보 3] 성공 x 구종 (9분할)')
print('=' * 82)
pnames = {0: '직구', 1: '변화구', 2: '오프'}
print(f'{"성공유형":<8}{"구종":<8}{"n":>10}{"전체비중":>10}')
for sn, sm in snames.items():
    for p in range(3):
        m = sm & (ptype == p)
        print(f'  {sn:<8}{pnames[p]:<8}{m.sum():>10,}{m.mean()*100:>9.2f}%')

print('\n' + '=' * 82)
print('[신규후보 4] 성공 x 압박(RISP) / 이닝후반')
print('=' * 82)
risp = ((r2 > 0) | (r3 > 0))
late = inning >= 7
for sn, sm in snames.items():
    a = sm & risp; b = sm & ~risp
    c = sm & late; d = sm & ~late
    print(f'  {sn:<8} RISP {a.sum():>8,} ({a.mean()*100:4.2f}%) / 무주자 {b.sum():>8,}   '
          f'후반 {c.sum():>8,} / 전반 {d.sum():>8,}')

print('\n' + '=' * 82)
print('[분할축 유망도 판정] 각 축이 성공을 얼마나 "다르게" 나누는가')
print('=' * 82)
print('  (성공행 안에서 그 축의 분산이 클수록 = 성공이 이질적이라는 증거)')
succ_rows = np.flatnonzero(succ)
for nm, arr in [('판정축(ball/strk/play)', np.where(ball[succ_rows] > 0.5, 0,
                                                    np.where(strike[succ_rows] > 0.5, 1, 2))),
                ('카운트유리도', cnt_adv[succ_rows]),
                ('구종', ptype[succ_rows]),
                ('RISP', risp[succ_rows].astype(int)),
                ('이닝후반', late[succ_rows].astype(int)),
                ('타자손', bhand[succ_rows])]:
    vals, cnts = np.unique(arr[arr >= 0], return_counts=True)
    p = cnts / cnts.sum()
    ent = -np.sum(p * np.log(p + 1e-12)) / np.log(len(p)) if len(p) > 1 else 0
    print(f'  {nm:<22} 분할수={len(p)}  정규화엔트로피={ent:.3f}  '
          f'최대집단비중={p.max()*100:.1f}%')
