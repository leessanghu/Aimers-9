"""trackman 물리량 추세: 실제로 공이 달라졌나, 아니면 라벨만 달라졌나?
구속/회전/무브먼트가 그대로인데 reverse/middle만 늘었다면 -> 측정/정의 변화 의심
구속이 올랐다면 -> 실제 투구 스타일 변화 (구속-제구 트레이드오프)"""
import numpy as np, pandas as pd, sys, time
sys.stdout.reconfigure(encoding='utf-8')
t0 = time.time()
def log(m): print(f'[{time.time()-t0:5.0f}s] {m}', flush=True)

cols = ['season', 'pitcher_trackman_id', 'pitch_type_group', 'rel_speed', 'spin_rate',
        'induced_vert_break', 'horz_break', 'extension', 'rel_height', 'rel_side', 'zone_speed']
tm = pd.read_csv('data/trackman_history.csv', encoding='utf-8-sig', usecols=cols)
log(f'trackman 로드 {len(tm):,}')

print('=== 시즌별 물리량 평균 ===')
g = tm.groupby('season')[['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break',
                          'extension', 'rel_height', 'rel_side', 'zone_speed']].mean()
print(g.round(3).to_string())
print()
print('2019 대비 변화:')
print((g - g.loc[2019]).round(3).to_string())
print()

print('=== 구종 구성비 (pitch_type_group) ===')
mix = tm.groupby('season')['pitch_type_group'].value_counts(normalize=True).unstack().fillna(0)
print(mix.round(4).to_string())
print()

print('=== 패스트볼만: 구속 추세 (구종믹스 효과 제거) ===')
fb = tm[tm.pitch_type_group == 'fastball']
print(fb.groupby('season')[['rel_speed', 'spin_rate', 'induced_vert_break', 'horz_break']].mean().round(3).to_string())
print()

print('=== 같은 투수 내 구속 변화 (개인 고정) ===')
py = fb.groupby(['pitcher_trackman_id', 'season'])['rel_speed'].agg(['mean', 'count']).reset_index()
py = py[py['count'] >= 100]
for s in range(2020, 2025):
    a = py[py.season == s - 1].set_index('pitcher_trackman_id')
    b = py[py.season == s].set_index('pitcher_trackman_id')
    common = a.index.intersection(b.index)
    if len(common) == 0:
        continue
    d = (b.loc[common, 'mean'] - a.loc[common, 'mean']).mean()
    league = fb[fb.season == s]['rel_speed'].mean() - fb[fb.season == s - 1]['rel_speed'].mean()
    print(f'  {s}: 공통투수={len(common):3d}  개인평균구속변화={d:+.3f}km/h  리그변화={league:+.3f}')
print()

print('=== 결측/커버리지 변화 (측정 인프라 변화 탐지) ===')
cover = tm.groupby('season').agg(
    n=('rel_speed', 'size'),
    speed_na=('rel_speed', lambda s: s.isna().mean()),
    spin_na=('spin_rate', lambda s: s.isna().mean()),
    ivb_na=('induced_vert_break', lambda s: s.isna().mean()),
    ext_na=('extension', lambda s: s.isna().mean()),
    zone_na=('zone_speed', lambda s: s.isna().mean()),
)
print(cover.round(4).to_string())
log('완료')
