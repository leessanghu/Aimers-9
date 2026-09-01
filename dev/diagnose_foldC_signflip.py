"""fold C만 rho가 양수인 원인 규명.

feat = sum_t P(t|count) * rate(pitcher, t)
이걸 성분분해:
  feat_count   = sum_t P(t|count) * global_rate(t)      <- 카운트만 변함(투수 무관)
  feat_pitcher = sum_t global_mix(t) * rate(pitcher,t)   <- 투수만 변함(카운트 무관)
  feat_inter   = feat - feat_count - feat_pitcher + const
각 성분의 rho를 fold별로 재면 어느 축에서 부호가 뒤집히는지 알 수 있다.

추가 진단:
  - fold별 train->valid 전역성공률 드리프트 (축소prior 편향의 크기)
  - fold별 학습창 길이 (투수x구종 셀당 표본수 = 축소 강도)
  - fold별 구종믹스 드리프트
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

meta = pd.read_parquet('dev/featcache_meta.parquet')
season = meta['season'].to_numpy()
y_all = meta['control_success'].to_numpy(np.float64)
v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
pid_all = df['pitcher_id'].to_numpy()
cs_all = (df['balls_before'].to_numpy() * 4 + df['strikes_before'].to_numpy())
ptype = np.load('dev/recovered_pitch_type.npy')


def build_blend(tag, heads4=False):
    if heads4:
        H = dict(
            base=avg([f'dev/phase90_cache/{tag}_base_{m}.npy' for m in ('d6', 'd8', 'sub')]),
            hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{m}.npy')) *
                            np.load(f'dev/phase90_cache/{tag}_snc_{m}.npy') for m in ('d6', 'd8')], axis=0),
            multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
            ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        )
    else:
        H = dict(
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
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    return np.clip(sum(W[k] * H[k] for k in H), 0, 1)


def rho_of(d, resid):
    dd = d - d.mean(); rr = resid - resid.mean()
    den = np.sqrt(np.mean(dd**2) * np.mean(rr**2))
    return float(np.mean(dd*rr) / den) if den > 1e-14 else 0.0


print('=' * 92)
print('[1] fold별 성분분해 rho')
print('=' * 92)
print(f'{"fold":<6}{"valid":<7}{"train창":<12}{"g(train)":>9}{"r(valid)":>9}{"드리프트":>9}'
      f'{"rho_전체":>10}{"rho_카운트":>11}{"rho_투수":>10}')

rows = []
for tag, upto, vs, h4 in [('A', 2023, 2024, False), ('B', 2022, 2023, True), ('C', 2021, 2022, False)]:
    va = season == vs
    yv = y_all[va]
    blend = build_blend(tag, heads4=h4)
    resid = yv - blend

    tr = (season <= upto) & (ptype >= 0)
    g = float(y_all[tr].mean())
    r_valid = float(yv.mean())

    # 통계 테이블
    mix_tab = pd.DataFrame({'cs': cs_all[tr], 't': ptype[tr]})
    mix_dist = mix_tab.groupby('cs')['t'].value_counts(normalize=True).unstack(fill_value=0)
    for t in range(3):
        if t not in mix_dist.columns:
            mix_dist[t] = 0.0
    mix_dist = mix_dist[[0, 1, 2]]
    global_mix = mix_tab['t'].value_counts(normalize=True).reindex([0, 1, 2]).fillna(0).to_numpy()
    ptab = pd.DataFrame({'p': pid_all[tr], 't': ptype[tr], 'y': y_all[tr]})
    p_rate = ptab.groupby(['p', 't'])['y'].agg(['sum', 'count'])
    K_SH = 60.0
    p_rate['rate'] = (p_rate['sum'] + K_SH * g) / (p_rate['count'] + K_SH)
    rate_wide = p_rate['rate'].unstack()
    for t in range(3):
        if t not in rate_wide.columns:
            rate_wide[t] = g
    rate_wide = rate_wide[[0, 1, 2]].fillna(g)
    global_rate_t = ptab.groupby('t')['y'].mean().reindex([0, 1, 2]).fillna(g).to_numpy()

    cs_va = cs_all[va]; pid_va = pid_all[va]
    mix_row = mix_dist.reindex(cs_va).fillna(pd.Series(global_mix, index=[0, 1, 2])).to_numpy(np.float64)
    rate_row = rate_wide.reindex(pid_va).fillna(g).to_numpy(np.float64)

    feat_full = (mix_row * rate_row).sum(axis=1)
    feat_count = (mix_row * global_rate_t).sum(axis=1)      # 투수 고정
    feat_pitch = (global_mix * rate_row).sum(axis=1)         # 카운트 고정

    rows.append((tag, vs, upto, g, r_valid, rate_wide, ptab))
    print(f'{tag:<6}{vs:<7}{f"<={upto}":<12}{g:>9.4f}{r_valid:>9.4f}{r_valid-g:>+9.4f}'
          f'{rho_of(feat_full, resid):>+10.5f}{rho_of(feat_count, resid):>+11.5f}'
          f'{rho_of(feat_pitch, resid):>+10.5f}')

print('\n' + '=' * 92)
print('[2] 투수x구종 셀 표본수 (축소강도 = 신호가 얼마나 살아남는가)')
print('=' * 92)
for tag, vs, upto, g, r_valid, rate_wide, ptab in rows:
    cell = ptab.groupby(['p', 't']).size()
    print(f'  fold{tag}(train<={upto}): 셀 중앙값={cell.median():.0f}  '
          f'p25={cell.quantile(.25):.0f}  p75={cell.quantile(.75):.0f}  '
          f'K=60 대비 셀중앙 비중={cell.median()/(cell.median()+60)*100:.0f}%  '
          f'rate 분산={rate_wide.to_numpy().var():.6f}')

print('\n' + '=' * 92)
print('[3] 연도별 실제 성공률 + 구종믹스 (2025가 어느 fold와 닮았는지 판단재료)')
print('=' * 92)
for yr in sorted(set(season.tolist())):
    m = season == yr
    mp = m & (ptype >= 0)
    print(f'  {int(yr)}: 성공률={y_all[m].mean():.4f}  '
          f'구종믹스 직구{np.mean(ptype[mp]==0)*100:.1f}% '
          f'변화구{np.mean(ptype[mp]==1)*100:.1f}% '
          f'오프{np.mean(ptype[mp]==2)*100:.1f}%')
