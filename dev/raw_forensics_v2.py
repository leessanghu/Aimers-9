"""raw 컬럼 통계 포렌식 (2차). 핵심 질문:
as-of 누적 카운터를 차분하면 per-pitch 라벨이 복원된다. 지금 우리가 복원해 쓰는 건
reverse/middle/ball 3개뿐. 복원 '가능한' 라벨을 전부 나열하고, 특히
  (1) pitchmix 카운터로 '이 투구의 구종'을 복원할 수 있는가?  <- 되면 완전 신규 채널
  (2) strike 라벨은? success/reverse/middle/ball/strike/other의 정확한 결합구조는?
  (3) 타자쪽(asof_batter_*) 카운터로 복원 가능한 건?
를 확인한다. 복원되는 라벨 하나하나가 multi-task 헤드의 새 y 후보다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
y = df['control_success'].to_numpy(np.float64)
print(f'행 {n:,}   투수 {df["pitcher_id"].nunique():,}   타자 {df["batter_id"].nunique():,}')

# ---------- 공통: 투수 시계열 차분 도구 ----------
pid = df['pitcher_id'].to_numpy()
order = df.sort_values(['pitcher_id', 'row_num']).index.to_numpy()
same_next = np.zeros(n, dtype=bool)
same_next[order[:-1]] = (pid[order][1:] == pid[order][:-1])
n_ = df['asof_pitcher_n'].fillna(0).to_numpy(np.float64)


def diff_counter(count_arr):
    """정렬된 순서에서 다음 행 - 현재 행. 마지막/투수경계는 nan."""
    c_ord = count_arr[order]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~same_next[order]] = np.nan
    out = np.empty(n); out[order] = d
    return out


def cnt_from_rate(col, n_col):
    return np.round(df[col].fillna(0).to_numpy(np.float64) * n_col)


# n이 정확히 1씩 증가하는 '연속' 행만 유효
dn = diff_counter(n_)
step = (dn == 1)
print(f'\nasof_pitcher_n이 정확히 +1 되는 연속행: {np.nansum(step):,.0f} ({np.nanmean(step)*100:.1f}%)')

print('\n' + '=' * 78)
print('[1] 투수쪽 per-pitch 라벨 복원 가능성')
print('=' * 78)
PITCHER_RATE_COLS = ['asof_pitcher_success_rate', 'asof_pitcher_reverse_rate',
                     'asof_pitcher_middle_rate', 'asof_pitcher_ball_rate',
                     'asof_pitcher_strike_rate']
labs = {}
for col in PITCHER_RATE_COLS:
    c = cnt_from_rate(col, n_)
    d = diff_counter(c)
    valid = step & np.isfinite(d)
    vals = d[valid]
    uniq, cnts = np.unique(vals, return_counts=True)
    short = col.replace('asof_pitcher_', '').replace('_rate', '')
    labs[short] = np.where(valid, d, np.nan)
    dist = '  '.join(f'{int(u)}:{c/len(vals)*100:.2f}%' for u, c in zip(uniq, cnts) if abs(u) < 3)
    print(f'  {short:<10} 차분값분포 [{dist}]   유효 {valid.sum():,}')

print('\n  [검증] 복원된 success 라벨 == control_success 인가?')
rec_succ = labs['success']
ok = np.isfinite(rec_succ)
match = (rec_succ[ok] == y[ok]).mean()
print(f'    일치율 = {match*100:.4f}%  (100%면 복원식이 정확하다는 증거)')

print('\n  [결합구조] success/reverse/middle/ball/strike의 동시분포 (유효행)')
allok = np.all([np.isfinite(labs[k]) for k in labs], axis=0)
tab = pd.DataFrame({k: labs[k][allok] for k in labs})
print(f'    유효행 {allok.sum():,}')
combo = tab.groupby(list(tab.columns)).size().sort_values(ascending=False)
print(f'    실제 등장하는 조합 {len(combo)}가지 (상위 12개):')
for idx, c in combo.head(12).items():
    lbl = '  '.join(f'{k}={int(v)}' for k, v in zip(tab.columns, idx))
    print(f'      {lbl}   n={c:>9,} ({c/allok.sum()*100:5.2f}%)')

print('\n' + '=' * 78)
print('[2] 구종(pitchmix) 복원 가능성  <- 신규 채널 후보')
print('=' * 78)
mn = df['asof_pitcher_pitchmix_n'].fillna(0).to_numpy(np.float64)
dmn = diff_counter(mn)
u, c = np.unique(dmn[np.isfinite(dmn)], return_counts=True)
print(f'  pitchmix_n 차분 분포: ' + '  '.join(
    f'{int(a)}:{b/np.isfinite(dmn).sum()*100:.2f}%' for a, b in zip(u, c) if abs(a) < 3))
mstep = (dmn == 1)
print(f'  pitchmix_n이 +1 되는 행: {np.nansum(mstep):,.0f} ({np.nanmean(mstep)*100:.1f}%)')
for col in ['asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate']:
    c_ = cnt_from_rate(col, mn)
    d_ = diff_counter(c_)
    valid = mstep & np.isfinite(d_)
    vals = d_[valid]
    u2, c2 = np.unique(vals, return_counts=True)
    short = col.replace('asof_pitcher_', '').replace('_rate', '')
    dist = '  '.join(f'{int(a)}:{b/len(vals)*100:.2f}%' for a, b in zip(u2, c2) if abs(a) < 3)
    labs['pt_' + short] = np.where(valid, d_, np.nan)
    print(f'  {short:<10} 차분값분포 [{dist}]')

pt_ok = np.all([np.isfinite(labs['pt_' + k]) for k in ('fastball', 'breaking', 'offspeed')], axis=0)
if pt_ok.sum() > 0:
    s = (labs['pt_fastball'] + labs['pt_breaking'] + labs['pt_offspeed'])
    print(f'\n  [검증] 세 구종 차분의 합이 1인 행 비율 = '
          f'{np.nanmean(s[pt_ok] == 1)*100:.2f}%  (100%면 구종 완전복원)')
    print(f'  복원 커버리지: {pt_ok.sum():,}행 / 전체 {n:,} = {pt_ok.mean()*100:.1f}%')
    ptype = np.where(labs['pt_fastball'] == 1, 0,
                     np.where(labs['pt_breaking'] == 1, 1,
                              np.where(labs['pt_offspeed'] == 1, 2, -1)))
    ptype = np.where(pt_ok, ptype, -1)
    vv = ptype[ptype >= 0]
    print(f'  복원된 구종 분포: 직구 {np.mean(vv==0)*100:.1f}%  변화구 {np.mean(vv==1)*100:.1f}%'
          f'  오프스피드 {np.mean(vv==2)*100:.1f}%')
    print(f'\n  [핵심] 구종별 control_success 실제 비율:')
    for k, nm in [(0, '직구'), (1, '변화구'), (2, '오프스피드')]:
        msk = ptype == k
        print(f'    {nm:<8} n={msk.sum():>9,}   success={y[msk].mean():.4f}')
    np.save('dev/recovered_pitch_type.npy', ptype)
    print(f'  -> dev/recovered_pitch_type.npy 저장')

print('\n' + '=' * 78)
print('[3] 타자쪽 카운터 복원 가능성')
print('=' * 78)
bid = df['batter_id'].to_numpy()
border = df.sort_values(['batter_id', 'row_num']).index.to_numpy()
bsame = np.zeros(n, dtype=bool)
bsame[border[:-1]] = (bid[border][1:] == bid[border][:-1])
bn = df['asof_batter_n'].fillna(0).to_numpy(np.float64)


def bdiff(count_arr):
    c_ord = count_arr[border]
    d = np.empty(n); d[:-1] = c_ord[1:] - c_ord[:-1]; d[-1] = np.nan
    d[~bsame[border]] = np.nan
    out = np.empty(n); out[border] = d
    return out


dbn = bdiff(bn)
bstep = (dbn == 1)
print(f'  asof_batter_n이 +1 되는 행: {np.nansum(bstep):,.0f} ({np.nanmean(bstep)*100:.1f}%)')
for col in ['asof_batter_success_rate', 'asof_batter_middle_rate']:
    c_ = cnt_from_rate(col, bn)
    d_ = bdiff(c_)
    valid = bstep & np.isfinite(d_)
    vals = d_[valid]
    u2, c2 = np.unique(vals, return_counts=True)
    short = col.replace('asof_batter_', 'b_').replace('_rate', '')
    dist = '  '.join(f'{int(a)}:{b/len(vals)*100:.2f}%' for a, b in zip(u2, c2) if abs(a) < 3)
    print(f'  {short:<12} 차분값분포 [{dist}]   유효 {valid.sum():,}')
