"""codex persona_features.py의 아이디어(pitcher/batter_id를 압박/베이스아웃/이닝/
점수차/상대팀 등으로 잘게 조건부 축소)를 우리 환경에서 순수 pandas로 재구현.
codex의 pkl은 전혀 쓰지 않는다 - 코드(로직)만 참고하고 우리 데이터로 처음부터 재학습.

11개 persona spec(codex의 PERSONA_SPECS 활성셋과 동일) x [rate, delta, log_n,
reliability, uncertainty, vs_asof] = 66개 신규피처. Rule4 안전(연도별 누적만 사용,
expanding, walk-forward: fold별로 train<=upto 데이터로만 축소테이블 계산).

CatBoost로 162피처+66persona피처를 학습해서 fold A/C honest 검증(중심화+무절편+대조군).
"""
import sys, time
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np, pandas as pd, joblib
from catboost import CatBoostClassifier

t0 = time.time()
def log(m): print(f'[{time.time()-t0:6.0f}s] {m}', flush=True)

B = 0.249807

PERSONA_SPECS = [
    # (name, group_cols, parent_cols, prior, anchor)
    ("batter_vs_hand", ("batter_id", "pitcher_hand"), ("batter_id",), 45.0, "batter"),
    ("batter_count", ("batter_id", "persona_count"), ("batter_id",), 70.0, "batter"),
    ("pitcher_count", ("pitcher_id", "persona_count"), ("pitcher_id",), 55.0, "pitcher"),
    ("pitcher_pressure", ("pitcher_id", "persona_pressure"), ("pitcher_id",), 70.0, "pitcher"),
    ("pitcher_inning", ("pitcher_id", "persona_inning"), ("pitcher_id",), 60.0, "pitcher"),
    ("pitcher_opponent", ("pitcher_id", "batter_team_id"), ("pitcher_id",), 65.0, "pitcher"),
    ("pitcher_baseout", ("pitcher_id", "persona_baseout"), ("pitcher_id",), 90.0, "pitcher"),
    ("pitcher_month", ("pitcher_id", "game_month"), ("pitcher_id",), 65.0, "pitcher"),
    ("direct_matchup", ("pitcher_id", "batter_id"), ("pitcher_id",), 120.0, "pitcher"),
    ("team_count_hand", ("pitcher_team_id", "persona_count", "batter_hand"), ("pitcher_team_id",), 120.0, "pitcher"),
    ("batter_pressure", ("batter_id", "persona_pressure"), ("batter_id",), 85.0, "batter"),
]


def add_persona_keys(df):
    out = pd.DataFrame(index=df.index)
    balls = df["balls_before"].astype("int16")
    strikes = df["strikes_before"].astype("int16")
    out["persona_count"] = balls * 3 + strikes
    risp = ((df["runner_on_2b"] == 1) | (df["runner_on_3b"] == 1)).astype("int8")
    high_li = (df["li"] >= 1.5).astype("int8")
    late = (df["inning"] >= 7).astype("int8")
    out["persona_pressure"] = high_li + 2 * risp + 4 * late
    out["persona_inning"] = np.select(
        [df["inning"] <= 3, df["inning"] <= 6, df["inning"] <= 9], [0, 1, 2], default=3).astype("int8")
    out["persona_baseout"] = (
        df["runner_on_1b"].astype("int8") + 2 * df["runner_on_2b"].astype("int8")
        + 4 * df["runner_on_3b"].astype("int8") + 8 * df["outs_before"].astype("int8"))
    for c in ("pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
              "pitcher_team_id", "batter_team_id", "game_month", "season",
              "asof_pitcher_success_rate", "asof_batter_success_rate"):
        out[c] = df[c].to_numpy()
    return out


def build_persona_features_walkforward(keyed, target, upto, va_mask):
    """train<=upto 데이터로만 그룹/부모 히스토리 테이블을 만들고(연도 무관, 전체누적),
    va_mask 구간에 적용. Rule4: 미래데이터 미사용, 자기 행 제외 불필요(에폭 자체가
    train<=upto로 이미 검증연도와 분리됨)."""
    tr = (keyed["season"] <= upto).to_numpy()
    global_mean = float(target[tr].mean())
    feats = []
    for name, group_cols, parent_cols, prior, anchor in PERSONA_SPECS:
        gcols = list(group_cols); pcols = list(parent_cols)
        uniq_cols = list(dict.fromkeys(gcols + pcols))
        work = keyed.loc[tr, uniq_cols].copy()
        work["_y"] = target[tr]
        g = work.groupby(gcols, observed=True, sort=False)["_y"].agg(["sum", "count"])
        p = work.groupby(pcols, observed=True, sort=False)["_y"].agg(["sum", "count"])

        row = keyed.loc[va_mask, uniq_cols].copy()
        row = row.merge(g.rename(columns={"sum": "gsum", "count": "gn"}), on=gcols, how="left")
        row = row.merge(p.rename(columns={"sum": "psum", "count": "pn"}), on=pcols, how="left")
        for c in ("gsum", "gn", "psum", "pn"):
            row[c] = row[c].fillna(0.0)

        parent_rate = (row["psum"] + 100.0 * global_mean) / (row["pn"] + 100.0)
        rate = (row["gsum"] + prior * parent_rate) / (row["gn"] + prior)
        reliability = row["gn"] / (row["gn"] + prior)
        uncertainty = np.sqrt(np.clip(rate * (1 - rate) / (row["gn"] + prior + 1.0), 0, None))
        anchor_col = "asof_pitcher_success_rate" if anchor == "pitcher" else "asof_batter_success_rate"
        anchor_val = keyed.loc[va_mask, anchor_col].fillna(global_mean).to_numpy()

        prefix = f"persona_{name}"
        feats.append(pd.DataFrame({
            f"{prefix}_rate": rate.to_numpy(np.float32),
            f"{prefix}_delta": (rate - parent_rate).to_numpy(np.float32),
            f"{prefix}_log_n": np.log1p(row["gn"]).to_numpy(np.float32),
            f"{prefix}_reliability": reliability.to_numpy(np.float32),
            f"{prefix}_uncertainty": uncertainty.to_numpy(np.float32),
            f"{prefix}_vs_asof": (rate.to_numpy() - anchor_val).astype(np.float32),
        }, index=keyed.index[va_mask]))
    return pd.concat(feats, axis=1)


# ---------- 로드 ----------
X = pd.read_parquet('dev/featcache_X.parquet')
meta = pd.read_parquet('dev/featcache_meta.parquet')
y = meta['control_success'].to_numpy(np.float64)
season = meta['season'].to_numpy()
raw_all = pd.read_csv('data/train.csv', encoding='utf-8-sig')
keyed = add_persona_keys(raw_all)
keyed['season'] = season

v95 = joblib.load('submit/model/model_artifacts_v95.pkl')
FEAT_ORDER = list(v95['feature_order'])

CB_PARAMS = dict(iterations=1500, learning_rate=0.03, depth=7, l2_leaf_reg=6.0,
                  loss_function='Logloss', eval_metric='Logloss', verbose=0,
                  early_stopping_rounds=80, random_seed=42)


def train_eval(upto, vs, tag):
    tr_m = season <= upto
    va_m = season == vs
    log(f'[{tag}] persona 피처 생성...')
    pf_tr_within = None  # persona값은 walk-forward로 va에 대해서만 만들면 됨(학습엔 train기간 OOF 필요)
    # 학습용: train기간 자체에도 persona가 필요 -> train기간을 연도별 확장(walk-forward, 자기연도 제외)
    years = sorted(set(season[tr_m].tolist()))
    tr_pieces = []
    for yv_ in years[1:]:  # 첫 해는 히스토리 없음 -> 스킵(축소가 global_mean으로 수렴하므로 굳이 포함 안함)
        piece = build_persona_features_walkforward(keyed, y, yv_ - 1, season == yv_)
        tr_pieces.append((season == yv_, piece))
    va_persona = build_persona_features_walkforward(keyed, y, upto, va_m)

    # 학습 X: train기간 중 히스토리 있는 연도만(첫해 제외), persona 피처 결합
    tr_idx_mask = np.zeros(len(y), dtype=bool)
    for m_, _ in tr_pieces:
        tr_idx_mask |= m_
    X_tr_persona = pd.concat([p for _, p in tr_pieces], axis=0).sort_index()
    Xtr = pd.concat([X.loc[tr_idx_mask, FEAT_ORDER], X_tr_persona], axis=1)
    ytr = y[tr_idx_mask]
    seasontr = season[tr_idx_mask]
    Xva = pd.concat([X.loc[va_m, FEAT_ORDER], va_persona], axis=1)
    yva = y[va_m]

    w = 0.5 ** ((upto - seasontr) / 2.0)
    n = len(ytr)
    n_es = int(n * 0.92)
    log(f'[{tag}] 학습 n={n:,}  persona피처={X_tr_persona.shape[1]}개  전체피처={Xtr.shape[1]}개')
    m = CatBoostClassifier(**CB_PARAMS)
    m.fit(Xtr.iloc[:n_es], ytr[:n_es], sample_weight=w[:n_es],
          eval_set=(Xtr.iloc[n_es:], ytr[n_es:]))
    p = np.clip(m.predict_proba(Xva)[:, 1], 0, 1)
    log(f'[{tag}] 학습완료 best_iter={m.best_iteration_}')

    sc = lambda pp: 1e5 * (1 - np.mean((np.clip(pp, 0, 1) - yva) ** 2) / B)
    print(f'\n=== fold {tag} (train<={upto} -> {vs}) ===')
    print(f'  persona_head 단독 BSS = {sc(p):.2f}')
    return p


log('=== fold A (train<=2023 -> 2024) ===')
p_A = train_eval(2023, 2024, 'A')
np.save('dev/cache_persona_A.npy', p_A)

log('=== fold C (train<=2021 -> 2022) ===')
p_C = train_eval(2021, 2022, 'C')
np.save('dev/cache_persona_C.npy', p_C)

log('v88_final 대비 클린검증...')
blend = np.load('dev/cache_v88_final_2024.npy')
yv = y[season == 2024]
Xv = X.loc[season == 2024]
mth = Xv['game_month'].to_numpy()
H1 = mth <= 6; H2 = ~H1
resid = yv - blend
sc2 = lambda pp, msk: 1e5 * (1 - np.mean((np.clip(pp[msk], 0, 1) - yv[msk]) ** 2) / B)
d = p_A - blend
rng = np.random.RandomState(8)
ctrl = rng.normal(0, d.std(), len(yv))


def honest(dd):
    gains = []
    for fit_m, ev_m in [(H1, H2), (H2, H1)]:
        mdf = dd[fit_m].mean(); mrf = resid[fit_m].mean()
        cv = np.mean((dd[fit_m]-mdf)*(resid[fit_m]-mrf))
        vr = np.mean((dd[fit_m]-mdf)**2)
        a = cv/vr if vr > 1e-14 else 0.0
        bl = blend.copy()
        bl[ev_m] = blend[ev_m] + a*(dd[ev_m]-mdf)
        gains.append(sc2(bl, ev_m) - sc2(blend, ev_m))
    return gains


gc = honest(ctrl)
g = honest(d)
print(f'\n=== v88_final 대비 클린 max-gain (fold A) ===')
print(f'  대조군  H1->H2={gc[0]:+7.2f}  H2->H1={gc[1]:+7.2f}  평균={np.mean(gc):+7.2f}')
print(f'  persona_head  H1->H2={g[0]:+7.2f}  H2->H1={g[1]:+7.2f}  평균={np.mean(g):+7.2f}')
log('완료')
