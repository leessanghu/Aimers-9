"""base 헤드(HGB, 현재 l2=5.0/depth 6&8&sub 평균)의 정규화를 풀면(L2 낮추고 depth 올리면)
fold A(train<=2023 -> 2024) 정직검증에서 실제로 좋아지는지 검증.
같은 X(162피처), 같은 tr_m/w, 같은 v88_final 조합식을 그대로 쓰고 base 헤드만 교체."""
import os, sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

import batter_split as bsplit
from batterform import K_BATTER, build_batter_table, transform_batter
from career_volatility import K_VOL, build_volatility_table, transform_volatility
from count_split import K_COUNT, build_count_table, transform_count
from crosses import add_crosses
from features import FeatureBuilder, TARGET_COL
from formfeat import build_role_table, transform_form, transform_role
from inning_split import K_INNING, build_inning_offset, build_inning_table, transform_inning
from inseason import build_season_end_table, transform_inseason, _pivots_from_table
from inseason_full import build_global_priors, build_season_end_table_full, transform_inseason_full
from lastyear import build_global_rates, build_lastyear_table, transform_lastyear
from pitchtype import build_matched, build_pitchtype_tables, transform_pitchtype
from platoon import K_PLATOON, build_platoon_table, transform_platoon
from trackman_profile import add_lown_interactions, build_trackman_profile, transform_trackman

TM_CACHE = "phase64_trackman_profile.parquet"
CD = "phase90_cache"
os.makedirs(CD, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

log("데이터 로드 + 피처 재구성 (v28/v29와 동일 162개)...")
df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
y = df[TARGET_COL].to_numpy()
g = float(df[TARGET_COL].mean())
sr = sorted(df["season"].unique().tolist())

fb = FeatureBuilder(seed=42, include_raw_rates=False, team_te_mode="expanding").fit(df)
X_base = fb.transform_train_oof(df).reset_index(drop=True)
se = build_season_end_table(df)
X_ins = transform_inseason(df, se, g, sr).reset_index(drop=True)
piv = _pivots_from_table(se, sr)
idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
prior = pd.Series(piv["rate"].reindex(idx).to_numpy()).fillna(g).to_numpy(np.float64)
X_plt = transform_platoon(df, build_platoon_table(df), prior, sr, k=K_PLATOON).reset_index(drop=True)
it, io = build_inning_table(df), build_inning_offset(df)
X_inn = transform_inning(df, it, io, prior, sr, k=K_INNING).reset_index(drop=True)
X_cnt = transform_count(df, build_count_table(df), prior, sr, k=K_COUNT).reset_index(drop=True)
X_pt = transform_pitchtype(df, build_pitchtype_tables(build_matched(df), sr), prior, g, sr).reset_index(drop=True)
X_ly = transform_lastyear(df, build_lastyear_table(df), build_global_rates(df), sr, k=30.0).reset_index(drop=True)
X_vol = transform_volatility(df, build_volatility_table(se), sr, k=K_VOL).reset_index(drop=True)
role_tbl = build_role_table(df)
X_role = transform_role(df, role_tbl, sr).reset_index(drop=True)
base_middle = np.full(len(df), float(df["asof_pitcher_middle_rate"].mean(skipna=True)))
X_form = transform_form(df, X_role, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                        base_middle).reset_index(drop=True)
prof = pd.read_parquet(TM_CACHE) if os.path.exists(TM_CACHE) else build_trackman_profile()
X_tm = transform_trackman(df, prof, sr).reset_index(drop=True)
lown_thr = float(df["asof_pitcher_n"].fillna(0).median())
X_tmx = add_lown_interactions(X_tm, df["asof_pitcher_n"].to_numpy(np.float64), lown_thr).reset_index(drop=True)
X_bat = transform_batter(df, build_batter_table(df), sr, g, k=K_BATTER).reset_index(drop=True)
n_end_row = np.nan_to_num(piv["N_end"].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
X_isf = transform_inseason_full(df, build_season_end_table_full(df), build_global_priors(df), sr,
                                n_end_row, X_ins["inseason_success_smooth"].to_numpy(np.float64),
                                X_ins["inseason_reverse_smooth"].to_numpy(np.float64)).reset_index(drop=True)
g_bmid = float(df["asof_batter_middle_rate"].mean(skipna=True))
X_bmid = bsplit.transform_batter_middle(df, bsplit.build_batter_middle_table(df), sr, g_bmid).reset_index(drop=True)
bmarg = bsplit.build_batter_marginal(df)
b_prior = bsplit.lookup_batter_prior(df, bmarg, sr, g)
X_bplat = bsplit.transform_bplatoon(df, bsplit.build_bplatoon_table(df), b_prior, sr,
                                    k=bsplit.K_BPLATOON).reset_index(drop=True)

X = pd.concat([X_base, X_ins, X_plt, X_inn, X_pt], axis=1).astype(np.float64)
C = add_crosses(X)
X = pd.concat([X, C, X_ly, X_cnt, X_vol, X_role, X_form, X_tm, X_tmx, X_bat,
               X_isf, X_bmid, X_bplat], axis=1).astype(np.float64)
log(f"피처 {X.shape[1]}개")

seasons = df["season"].to_numpy(np.float64)


def recency_weight(seasons_, half_life=2.0, ref=None):
    r = ref if ref is not None else seasons_.max()
    return 0.5 ** ((r - seasons_) / half_life)


tr_m = seasons <= 2023
va_m = seasons == 2024
yv = y[va_m].astype(np.float64)
Xva = X.loc[va_m]
w = recency_weight(seasons, 2.0, ref=2023)

BASE_HGB = dict(max_iter=500, learning_rate=0.03, l2_regularization=5.0, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=20)

# 기존 3변종 (l2=5.0, 얕음) 캐시 로드
orig_preds = []
for n in ("d6", "d8", "sub"):
    orig_preds.append(np.load(f"{CD}/A_base_{n}.npy"))
p_base_orig = np.mean(orig_preds, axis=0)

# 정규화 완화 변종: L2 대폭 낮추고 depth/leaf 키움
DEREG_VARIANTS = [
    ("dereg_d10", dict(max_depth=10, max_leaf_nodes=63, l2_regularization=1.0, random_state=42)),
    ("dereg_d12sub", dict(max_depth=12, max_leaf_nodes=127, l2_regularization=1.0,
                          max_features=0.6, random_state=123)),
]


def fit_cached(tag, extra):
    f = f"{CD}/A_base_{tag}.npy"
    if os.path.exists(f):
        log(f"    {tag} 캐시")
        return np.load(f)
    p = dict(BASE_HGB)
    p.update(extra)
    ts = time.time()
    m = HistGradientBoostingClassifier(**p).fit(X.loc[tr_m], y[tr_m].astype(np.float64), sample_weight=w[tr_m])
    out = m.predict_proba(Xva)[:, 1]
    np.save(f, out)
    log(f"    {tag} 완료 iters={m.n_iter_} ({time.time()-ts:.0f}s)")
    return out


log("정규화 완화 변종 학습...")
dereg_preds = [fit_cached(n, e) for n, e in DEREG_VARIANTS]
p_base_dereg = np.mean(dereg_preds, axis=0)

unc = 0.249807
def sc(p_): return 1e5 * (1 - np.mean((np.clip(p_, 0, 1) - yv) ** 2) / unc)

print()
print(f"base(원래, l2=5.0, depth6/8/sub)  score={sc(p_base_orig):.2f}  std={p_base_orig.std():.4f}  범위=[{p_base_orig.min():.3f},{p_base_orig.max():.3f}]")
print(f"base(완화, l2=1.0, depth10/12sub) score={sc(p_base_dereg):.2f}  std={p_base_dereg.std():.4f}  범위=[{p_base_dereg.min():.3f},{p_base_dereg.max():.3f}]")
print(f"base 헤드 단독 델타 = {sc(p_base_dereg)-sc(p_base_orig):+.2f}")

# v88_final 조합에서 base만 교체해서 비교
v88 = __import__("joblib").load("../submit/model/model_artifacts_v88.pkl")
W = dict(base=v88['base_weight'], hurdle=v88['hurdle_weight'], multires=v88['multires_weight'],
         ordinal=v88['ordinal_weight'], midother=v88['midother_weight'], condball=v88['condball_weight'],
         countresid=v88['countresid_weight'], future50=v88['future50_weight'], mc5=v88['mc5_weight'],
         ingame=v88['ingame_weight'])
avg = lambda ps: np.mean([np.load(q) for q in ps], axis=0)
p = 'A'
H = dict(
    hurdle=np.mean([(1 - np.load(f'{CD}/{p}_core_{n}.npy')) * np.load(f'{CD}/{p}_snc_{n}.npy') for n in ('d6', 'd8')], axis=0),
    multires=avg([f'idea13_cache/{p}_multires_s{s}.npy' for s in (42, 7)]),
    ordinal=avg([f'idea13_cache/{p}_ordinal_s{s}.npy' for s in (42, 7)]),
    midother=avg([f'idea46_cache/{p}_midother_s{s}.npy' for s in (42, 7)]),
    condball=avg([f'idea54_cache/{p}_cond_ball_s{s}.npy' for s in (42, 7)]),
    countresid=avg([f'idea54_cache/{p}_count_resid_s{s}.npy' for s in (42, 7)]),
    future50=avg([f'idea54_cache/{p}_future50_multi_s{s}.npy' for s in (42, 7)]),
)
P11 = np.load('idea75_cache/A_proba11.npy')
H['mc5'] = np.clip(P11 @ np.asarray(v88['mc5_succ'], dtype=np.float64), 0, 1)
ing = np.load('idea80_cache/A_ingame_heads_s42.npy')
H['ingame'] = np.clip(ing[:, 0] if ing.ndim > 1 else ing, 0, 1)
risk = P11[:, [9, 10]].sum(axis=1)
cut = np.maximum(0.0, risk - float(v88['risk_thr']))

def build_final(p_base_):
    H2 = dict(H); H2['base'] = p_base_
    raw = sum(W[k] * H2[k] for k in H2)
    return np.clip(raw - float(v88['risk_alpha']) * (cut - float(v88['risk_center'])) + float(v88['level_shift']), 0, 1)

final_orig = build_final(p_base_orig)
final_dereg = build_final(p_base_dereg)
print()
print(f"v88_final(base=원래)  score={sc(final_orig):.2f}")
print(f"v88_final(base=완화)  score={sc(final_dereg):.2f}")
print(f"전체 델타 = {sc(final_dereg)-sc(final_orig):+.2f}")

# 완화판 캘리브레이션 곡선 (조밀)
order = np.argsort(final_dereg)
pred_s = final_dereg[order]; y_s = yv[order]
n = len(yv)
for K in (100,):
    edges_idx = np.linspace(0, n, K + 1).astype(int)
    flagged = []
    for i in range(K):
        s_, e_ = edges_idx[i], edges_idx[i + 1]
        if e_ <= s_:
            continue
        p_bin = pred_s[s_:e_]; y_bin = y_s[s_:e_]
        m_pred = p_bin.mean(); m_act = y_bin.mean(); cnt = e_ - s_
        se_ = np.sqrt(max(m_act * (1 - m_act), 1e-9) / cnt)
        gap = m_act - m_pred
        z = gap / se_ if se_ > 0 else 0.0
        if abs(z) > 3:
            flagged.append((m_pred, m_act, cnt, gap, z))
    print(f"\n[완화판] Reliability curve {K}구간: |z|>3 어긋남 {len(flagged)}/{K}")
    for m_pred, m_act, cnt, gap, z in flagged[:20]:
        print(f"   pred={m_pred:.4f} actual={m_act:.4f} n={cnt:,} gap={gap:+.4f} z={z:+.1f}")
print(f"완화판 예측 std={final_dereg.std():.4f}  (원래={final_orig.std():.4f})")
log("완료")
