"""count_state==2(0-2 카운트, fold A/C에서 가장 판별력이 약한 구간) 전용
투수별 EB축소 슬로프. k2slope(strikes_before==2 전체, 실측+0.83)와 동일 레시피를
0-2 카운트만 특정해서 재구성 - 더 국소적인 신호를 노림.

검증 3종 세트(오늘 확립):
  1) 대조군(d=0, 랜덤노이즈) 대비 유의한지
  2) 중심화 + 절편없음 H1/H2
  3) fold A와 fold C 양쪽 재현
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd

K_CONST = 1e5 / 0.249807
meta = pd.read_parquet('dev/featcache_meta.parquet')
X = pd.read_parquet('dev/featcache_X.parquet')
season = meta['season'].to_numpy()
y = meta['control_success'].to_numpy(np.float64)
raw = pd.read_csv('data/train.csv', encoding='utf-8-sig', usecols=['season', 'pitcher_id', 'balls_before', 'strikes_before'])
count_state = (raw['balls_before'] * 4 + raw['strikes_before']).to_numpy(np.float64)
pid = raw['pitcher_id'].to_numpy()
is02 = (count_state == 2).astype(np.float64)

blend_cache = {}
try:
    blend_cache['A'] = np.load('dev/cache_v88_final_2024.npy')
except Exception:
    pass


def build_gap_table(train_mask, K_shrink):
    """train 구간에서 (0-2 카운트 성공률 - 그 투수 전체 성공률) 을 표본크기로 축소해 pitcher_id -> gap, n."""
    df = pd.DataFrame({'pid': pid[train_mask], 'is02': is02[train_mask], 'y': y[train_mask]})
    overall = df.groupby('pid')['y'].mean()
    on02 = df[df['is02'] == 1].groupby('pid').agg(y02=('y', 'mean'), n02=('y', 'size'))
    tab = on02.join(overall.rename('y_all'), how='left')
    gap_raw = tab['y02'] - tab['y_all']
    shrunk = gap_raw * (tab['n02'] / (tab['n02'] + K_shrink))
    return shrunk.to_dict(), tab['n02'].to_dict()


def apply_table(gap_by_pid, target_mask):
    g = np.array([gap_by_pid.get(p, 0.0) for p in pid[target_mask]], dtype=np.float64)
    is02_t = is02[target_mask]
    return is02_t * g  # 0-2 카운트인 행에만 적용, 아니면 0


def eval_fold(tag, upto, vs, K_shrink=200.0):
    tr = season <= upto
    va = season == vs
    yv = y[va]
    if tag in blend_cache:
        blend = blend_cache[tag]
    else:
        print(f'  [fold {tag}] cache_v88_final 없음 - 스킵')
        return None
    gap_tab, n_tab = build_gap_table(tr, K_shrink)
    d_full = apply_table(gap_tab, va)  # d = 신호 (아직 alpha 안 곱함)

    resid = blend - yv
    sc = lambda p, m: 1e5 * (1 - np.mean((np.clip(p[m], 0, 1) - yv[m]) ** 2) / 0.249807)
    mth = X.loc[va, 'game_month'].to_numpy()
    H1 = mth <= 6
    H2 = ~H1
    allm = np.ones(len(yv), bool)

    def honest(d):
        gains, coefs = [], []
        for fit_m, ev_m in [(H1, H2), (H2, H1)]:
            mdf = d[fit_m].mean()
            mrf = resid[fit_m].mean()
            cv = np.mean((d[fit_m] - mdf) * (resid[fit_m] - mrf))
            vr = np.mean((d[fit_m] - mdf) ** 2)
            a = cv / vr if vr > 1e-14 else 0.0
            bl = blend.copy()
            bl[ev_m] = blend[ev_m] + a * (d[ev_m] - mdf)
            gains.append(sc(bl, ev_m) - sc(blend, ev_m))
            coefs.append(a)
        return gains, coefs

    rng = np.random.RandomState(1)
    g_ctrl, _ = honest(rng.normal(0, 0.02, len(yv)))
    g_real, coefs = honest(d_full)

    n02 = (count_state[va] == 2).sum()
    print(f'  [fold {tag}] 0-2카운트 적용행={n02:,}/{len(yv):,} ({n02/len(yv)*100:.1f}%)')
    print(f'    대조군(랜덤)  H1->H2={g_ctrl[0]:+7.2f}  H2->H1={g_ctrl[1]:+7.2f}  평균={np.mean(g_ctrl):+7.2f}')
    print(f'    0-2슬로프    H1->H2={g_real[0]:+7.2f}  H2->H1={g_real[1]:+7.2f}  평균={np.mean(g_real):+7.2f}  '
          f'a(H1)={coefs[0]:+.4f} a(H2)={coefs[1]:+.4f}')
    return np.mean(g_real), np.mean(g_ctrl)


print('=== 0-2 카운트 투수별 슬로프 검증 ===')
print('\n[fold A: train<=2023 -> 2024]')
res_a = eval_fold('A', 2023, 2024)

print('\n[fold C: train<=2021 -> 2022] (blend 캐시 없어 8헤드 근사로 재구성)')
import joblib
v88 = joblib.load('submit/model/model_artifacts_v88.pkl')
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)


def build8(tag):
    return dict(
        base=avg([f'dev/phase90_cache/{tag}_base_{n}.npy' for n in ('d6', 'd8', 'sub')]),
        hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{n}.npy')) * np.load(f'dev/phase90_cache/{tag}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
        multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
        ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        midother=avg([f'dev/idea46_cache/{tag}_midother_s{s}.npy' for s in (42, 7)]),
        condball=avg([f'dev/idea54_cache/{tag}_cond_ball_s{s}.npy' for s in (42, 7)]),
        countresid=avg([f'dev/idea54_cache/{tag}_count_resid_s{s}.npy' for s in (42, 7)]),
        future50=avg([f'dev/idea54_cache/{tag}_future50_multi_s{s}.npy' for s in (42, 7)]),
    )


H = build8('C')
W = {k: float(v88[f'{k}_weight']) for k in H}
t = sum(W.values())
W = {k: v / t for k, v in W.items()}
blend_cache['C'] = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
res_c = eval_fold('C', 2021, 2022)

print('\n=== 종합 판정 ===')
if res_a and res_c:
    print(f'  fold A 평균이득={res_a[0]:+.2f} (대조군 {res_a[1]:+.2f})')
    print(f'  fold C 평균이득={res_c[0]:+.2f} (대조군 {res_c[1]:+.2f})')
    ok = res_a[0] > 0 and res_c[0] > 0 and res_a[0] > abs(res_a[1]) * 3 and res_c[0] > abs(res_c[1]) * 3
    print(f'  두 fold 모두 양수 & 대조군 압도 = {ok}')
