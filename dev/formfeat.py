"""투수의 '현재 폼' 피처 + 선발/불펜 역할 피처.

배경 (IDEA.md 측정치):
  prev5의 자기 베이스라인 대비 편차 SD = 0.0579. 표본잡음(5경기 ~250구 -> SD~0.032)을 걷어내면
  진짜 폼 변동 SD ~= 0.048. 이건 투수 실력 개인차 SD(0.0555)에 맞먹는 크기다.

그런데 현재 모델은 asof_pitcher_prev{1,3,5}_game_* 를 이렇게 쓰고 있다:
  - 원본 확률값 그대로 (beta smoothing 없음 -> 소표본 노이즈 그대로 유입)
  - 결측은 '리그 전역 평균'으로 채움 (그 투수 자신의 수준이 아니라!)
  - 차분은 확률 공간 raw difference 2개뿐 (x_prev5_minus_career, x_prev1_minus_prev5)

여기서는 세 가지를 고친다:
  (1) 베이스라인을 리그평균이 아니라 '그 투수 자신의 현재 실력 추정치'로
  (2) 표본 신뢰도(추정 투구수)만큼 베이스라인 쪽으로 축소
  (3) logit 공간에서 차분 (확률 0.5 근처와 0.8 근처의 같은 차이는 의미가 다르다)

표본 신뢰도가 필요한데 prev_k의 분모(투구수)는 주어지지 않는다. hidden_denominator로 역산하는 건
분모 모호성(65~69%) 때문에 실패했으므로, 대신 '그 투수의 등판당 평균 투구수'를 train에서
정적으로 추정해서 est_n = ppa * k 로 쓴다. 이 값은 역할(선발/불펜) 피처로도 그대로 쓰인다.

규칙 준수: 역할 프로파일은 train에서만 만들고 (pitcher, season-1) 조회. 폼 피처는 공식 asof_*
컬럼의 행 내부 변환. test 행 간 참조 없음.
"""

import numpy as np
import pandas as pd

EPS = 1e-6
K_FORM = 40.0        # 폼 추정치를 베이스라인 쪽으로 축소하는 강도 (투구수 단위)
K_ROLE = 3.0         # 역할 프로파일 축소 (등판 수 단위)

ROLE_COLS = ["role_ppa", "role_first_inn_share", "role_late_share", "role_med_inning", "role_n_app"]
FORM_COLS = [
    "form1_success", "form3_success", "form5_success",
    "form1_middle", "form3_middle", "form5_middle",
    "form_accel", "form_1_minus_3", "form_3_minus_5",
    "form_reliability", "form_missing",
]


def _logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


# ======================================================================
# 역할 프로파일 — train 행 시퀀스에서 '등판'을 복원해 등판당 투구수/이닝 분포를 낸다
# ======================================================================

def build_role_table(df):
    """(pitcher_id, season) -> 역할 프로파일.

    등판 경계 추정: 같은 투수의 행을 시간순(row_num)으로 보며
    (season, game_month, game_dayofweek)가 바뀌거나 inning이 감소하면 새 등판으로 본다.
    train.csv에 game_id가 없어서 쓰는 근사지만, 선발/불펜 구분에는 충분하다."""
    d = df[["pitcher_id", "season", "game_month", "game_dayofweek", "inning", "row_num"]].copy()
    d = d.sort_values(["pitcher_id", "row_num"])

    pid = d["pitcher_id"].to_numpy()
    key = (d["season"].to_numpy() * 10000 + d["game_month"].to_numpy() * 100
           + d["game_dayofweek"].to_numpy())
    inn = d["inning"].to_numpy()

    new_p = np.empty(len(d), dtype=bool)
    new_p[0] = True
    new_p[1:] = pid[1:] != pid[:-1]

    new_k = np.empty(len(d), dtype=bool)
    new_k[0] = True
    new_k[1:] = key[1:] != key[:-1]

    drop = np.empty(len(d), dtype=bool)
    drop[0] = False
    drop[1:] = inn[1:] < inn[:-1]

    d["_app"] = np.cumsum(new_p | new_k | drop)

    app = d.groupby(["pitcher_id", "season", "_app"]).agg(
        pitches=("inning", "size"), first_inn=("inning", "min"), med_inn=("inning", "median"))
    app = app.reset_index()

    tbl = app.groupby(["pitcher_id", "season"]).agg(
        role_n_app=("pitches", "size"),
        role_ppa=("pitches", "mean"),
        role_first_inn_share=("first_inn", lambda s: float((s <= 1).mean())),
        role_late_share=("med_inn", lambda s: float((s >= 8).mean())),
        role_med_inning=("med_inn", "median"),
    ).reset_index()
    return tbl


def _expanding_role(tbl, seasons_range):
    """시즌 누적 역할 프로파일 (등판 수 가중)."""
    rows = []
    for pid, grp in tbl.groupby("pitcher_id"):
        grp = grp.sort_values("season")
        n_cum = 0.0
        acc = {c: 0.0 for c in ROLE_COLS if c != "role_n_app"}
        for _, r in grp.iterrows():
            n = float(r["role_n_app"])
            for c in acc:
                v = r[c]
                if np.isfinite(v):
                    acc[c] += v * n
            n_cum += n
            out = {"pitcher_id": pid, "season": int(r["season"]), "role_n_app": n_cum}
            for c in acc:
                out[c] = acc[c] / n_cum if n_cum > 0 else np.nan
            rows.append(out)
    return pd.DataFrame(rows)


def export_stats(role_tbl, seasons_range, k=K_ROLE, k_form=K_FORM):
    return {"role_table": role_tbl, "seasons_range": list(seasons_range),
            "k_role": float(k), "k_form": float(k_form)}


def transform_role(df, role_tbl, seasons_range, k=K_ROLE):
    """각 행에 자기 투수의 season-1 시점 누적 역할 프로파일을 붙인다."""
    exp = _expanding_role(role_tbl, seasons_range)
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    glob = {c: float(exp[c].median()) for c in ROLE_COLS if c != "role_n_app"}

    piv_n = exp.pivot_table(index="pitcher_id", columns="season", values="role_n_app", aggfunc="first")
    piv_n = piv_n.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
    n_app = np.nan_to_num(piv_n.reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    out = pd.DataFrame(index=df.index)
    out["role_n_app"] = np.log1p(n_app)
    for c in ROLE_COLS:
        if c == "role_n_app":
            continue
        p = exp.pivot_table(index="pitcher_id", columns="season", values=c, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
        v = p.reindex(idx).to_numpy().astype(np.float64)
        gm = glob[c]
        v = np.where(np.isfinite(v), v, gm)
        out[c] = (n_app * v + k * gm) / (n_app + k)

    # 피로 상호작용: 같은 7회라도 선발(100구 넘김)과 불펜(방금 등판)은 정반대다.
    # 트리는 inning과 role을 각각은 볼 수 있어도 이 곱을 스스로 만들기 어렵다.
    out["role_x_inning"] = out["role_ppa"].to_numpy() * df["inning"].to_numpy(np.float64)
    return out.astype(np.float64)


# ======================================================================
# 폼 피처 — 공식 asof prev{1,3,5} 컬럼의 행 내부 변환
# ======================================================================

def transform_form(df, role_feats, baseline_success, baseline_middle, k_form=K_FORM):
    """
    baseline_success/middle: 각 행의 '그 투수 현재 실력 추정치' (inseason_success_smooth 등).
                             리그 평균이 아니라 반드시 투수 개인 값을 넣는다.
    role_feats: transform_role 결과 (role_ppa 사용 -> 표본 신뢰도 추정)
    """
    out = pd.DataFrame(index=df.index)
    ppa = np.clip(role_feats["role_ppa"].to_numpy(np.float64), 1.0, None)

    base_s = np.clip(np.asarray(baseline_success, dtype=np.float64), EPS, 1 - EPS)
    base_m = np.clip(np.asarray(baseline_middle, dtype=np.float64), EPS, 1 - EPS)
    lb_s, lb_m = _logit(base_s), _logit(base_m)

    miss = df["asof_pitcher_prev1_game_success_rate"].isna().to_numpy()
    out["form_missing"] = miss.astype(np.float64)

    forms = {}
    for k in (1, 3, 5):
        n_est = ppa * k                      # 그 투수의 등판당 투구수 x 경기 수
        for kind, base_p, lb in (("success", base_s, lb_s), ("middle", base_m, lb_m)):
            col = f"asof_pitcher_prev{k}_game_{kind}_rate"
            raw = df[col].to_numpy(np.float64)
            # 결측은 리그평균이 아니라 '그 투수 자신의 베이스라인'으로 (편차 0 = 정보 없음)
            raw = np.where(np.isfinite(raw), raw, base_p)
            # 신뢰도만큼만 반영: 표본이 적으면 베이스라인 쪽으로 끌어당긴다
            p_sm = (n_est * raw + k_form * base_p) / (n_est + k_form)
            f = _logit(p_sm) - lb
            out[f"form{k}_{kind}"] = f
            forms[(k, kind)] = f

    out["form_accel"] = forms[(1, "success")] - forms[(5, "success")]
    out["form_1_minus_3"] = forms[(1, "success")] - forms[(3, "success")]
    out["form_3_minus_5"] = forms[(3, "success")] - forms[(5, "success")]
    out["form_reliability"] = np.log1p(ppa)

    return out[FORM_COLS].astype(np.float64)
