"""전역 조건부 테이블 파생피처 (design_derived_features.md 설계).
투수별로 안 쪼개서 표본부족 회피, 전역대비 편차로 저장해서 드리프트 내성 확보.

피처 후보 (전부 학습데이터 train<=upto 통계 -> test는 count_state/구종만 참조, Rule4 안전):
  f1 = P(succ_ball|count) - global_succ_ball     : 존밖성공이 이 카운트에서 얼마나 잦은가(편차)
  f2 = P(succ_strk|count) - global_succ_strk
  f3 = P(wild|count) - global_wild               : 크게벗어남이 이 카운트에서 얼마나 잦은가
  f4 = P(succ_ball|count,구종) - global_succ_ball
  f5 = P(wild|구종) - global_wild                 : 구종만으로 조건화(카운트 무관)

fold A/C 3폴드(A,B,C) rho 부호일치 확인 + mc6/strk와 독립성(d상관) 확인.
피처는 XGB/CatBoost가 아니라 '기존 8헤드 블렌드에 직접 더하는' 형태가 아니라
'새 헤드의 입력피처'로 써야 하므로, 여기서는 먼저 순수 피처값 자체의 잔차상관만
빠르게 스크리닝(HGB 없이 그냥 편차값 자체를 후보로)하고, 통과하면 헤드를 만든다.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib

B_ = 0.249807
K = 1e5 / B_

df = pd.read_csv('data/train.csv', encoding='utf-8-sig')
df['row_num'] = df['row_id'].str.replace('TRAIN_', '', regex=False).astype(int)
n = len(df)
y = df['control_success'].to_numpy(np.float64)
season = df['season'].to_numpy()
pid = df['pitcher_id'].to_numpy()
cs = (df['balls_before'].to_numpy() * 4 + df['strikes_before'].to_numpy())
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
call = np.load('dev/recovered_call_axis.npy')
ball, strike, inplay = call[:, 0], call[:, 1], call[:, 2]
ptype = np.load('dev/recovered_pitch_type.npy')
valid = np.isfinite(rev) & np.isfinite(mid) & np.isfinite(ball)
nd = valid & (mid < 0.5) & (rev < 0.5)
wild = (nd & (y == 0)).astype(np.float64)
succ_ball = (nd & (y == 1) & (ball > 0.5)).astype(np.float64)
succ_strk = (nd & (y == 1) & (strike > 0.5)).astype(np.float64)
LABS = {'wild': (wild, valid), 'succ_ball': (succ_ball, nd & (y == 1) | (nd & (y == 0))),
        'succ_strk': (succ_strk, nd & (y == 1) | (nd & (y == 0)))}
# 정의역: succ_ball/succ_strk는 'nd 전체'(성공+실패) 중 그 사건 비율로 정의 (0 or 1)
dom_nd = nd

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


def cond_feat(target, domain, group_key, upto, va_mask):
    """train<=upto, domain==1인 행에서 group_key별 target 평균(=P(target|group,domain))
    - 전역평균, 편차로 반환. va_mask 행에 적용."""
    tr = (season <= upto) & (domain > 0.5) & np.isfinite(target)
    tab = pd.DataFrame({'g': group_key[tr], 't': target[tr]})
    grp = tab.groupby('g')['t'].mean()
    gmean = float(tab['t'].mean())
    out = pd.Series(group_key[va_mask]).map(grp).fillna(gmean).to_numpy(np.float64)
    return out - gmean


results = {}
for tag, upto, vs in [('A', 2023, 2024), ('B', 2022, 2023), ('C', 2021, 2022)]:
    va = season == vs
    yv = y[va]
    heads4 = tag == 'B'
    if heads4:
        H = dict(
            base=avg([f'dev/phase90_cache/{tag}_base_{m}.npy' for m in ('d6', 'd8', 'sub')]),
            hurdle=np.mean([(1 - np.load(f'dev/phase90_cache/{tag}_core_{m}.npy')) *
                            np.load(f'dev/phase90_cache/{tag}_snc_{m}.npy') for m in ('d6', 'd8')], axis=0),
            multires=avg([f'dev/idea13_cache/{tag}_multires_s{s}.npy' for s in (42, 7)]),
            ordinal=avg([f'dev/idea13_cache/{tag}_ordinal_s{s}.npy' for s in (42, 7)]),
        )
    else:
        H = build8(tag)
    W = {k: float(v95[f'{k}_weight']) for k in H}
    t_ = sum(W.values()); W = {k: v / t_ for k, v in W.items()}
    blend = np.clip(sum(W[k] * H[k] for k in H), 0, 1)
    resid = yv - blend
    E_r2 = float(np.mean(resid ** 2))

    f1 = cond_feat(succ_ball, dom_nd.astype(float), cs, upto, va)
    f2 = cond_feat(succ_strk, dom_nd.astype(float), cs, upto, va)
    f3 = cond_feat(wild, valid.astype(float), cs, upto, va)
    ptcs = cs.astype(str) + '_' + np.where(ptype >= 0, ptype, -1).astype(str)
    f4 = cond_feat(succ_ball, dom_nd.astype(float), ptcs, upto, va)
    f5 = cond_feat(wild, valid.astype(float), ptype, upto, va)

    print(f'\n=== fold {tag} ({vs}) ===')
    for nm, f in [('f1_succball|cnt', f1), ('f2_succstrk|cnt', f2), ('f3_wild|cnt', f3),
                  ('f4_succball|cnt,pt', f4), ('f5_wild|pt', f5)]:
        d = f - f.mean()
        V = float(np.mean(d ** 2))
        if V < 1e-14:
            print(f'  {nm:<20} 분산0, 스킵')
            continue
        A = float(np.mean(d * (blend - yv)))
        rho = -A / np.sqrt(V * E_r2)
        results.setdefault(nm, {})[tag] = rho
        print(f'  {nm:<20} std={np.sqrt(V):.5f}  rho={rho:+.5f}  최대이득(로컬,참고)={K*A**2/V:+6.2f}')

print(f'\n=== 종합 (3fold 부호일치 확인) ===')
print(f'{"피처":<20}{"foldA":>10}{"foldB":>10}{"foldC":>10}{"부호":>8}')
for nm, r in results.items():
    a, b, c = r.get('A', np.nan), r.get('B', np.nan), r.get('C', np.nan)
    signs = [np.sign(v) for v in (a, b, c) if not np.isnan(v)]
    agree = len(set(signs)) == 1
    print(f'{nm:<20}{a:>+10.5f}{b:>+10.5f}{c:>+10.5f}{"O" if agree else "X":>8}')
