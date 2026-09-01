"""범용 조건부 분할 피처 팩토리 — platoon.py / inning_split.py를 일반화.

검증된 구조를 그대로 파라미터화한다:
  (entity, context, season) 누적 -> pivot+ffill+stack -> (entity, context, season-1) 조회
  -> (그 entity의 직전시즌 실력 + 전역 context 효과)를 prior로 경험적 베이즈 축소
  -> 피처 = 축소된 값 - prior (순수 개인 상호작용) + log1p(셀 표본수)

전역 context 주효과를 prior에 넣어 빼는 이유: 모델은 이미 balls/strikes/month/team_id 등을
원시 피처로 갖고 있다. 안 빼면 주효과와 중복되어 개인 상호작용이 희석된다.

K는 phase12b에서 cov(직전편차, 다음시즌편차)= Var(안정적 편차)로 직접 유도한 값을 쓴다.
(관측분산-노이즈 방식은 시즌 공존 같은 교란에 취약해서 월/상대팀을 4~6배 과대평가했음)

leakage: 테이블/오프셋 전부 fit(=train)에서만 만들고, 각 행은 자기 entity의 '직전 시즌 끝'
까지 누적된 셀만 조회한다. 같은 시즌의 다른 행, test.csv의 다른 행은 전혀 참조하지 않는다.
"""

import numpy as np
import pandas as pd

TARGET = "control_success"


def build_cond_table(df, entity_col, ctx):
    """(entity, ctx, season) -> 그 시즌 끝까지의 누적 (성공수 s, 표본 n)."""
    d = pd.DataFrame({"_e": df[entity_col].to_numpy(), "_c": np.asarray(ctx),
                      "season": df["season"].to_numpy(), TARGET: df[TARGET].to_numpy()})
    g = d.groupby(["_e", "_c", "season"])[TARGET].agg(s="sum", n="count").sort_index()
    return g.groupby(level=[0, 1]).cumsum().reset_index()


def build_cond_offset(fit_df, ctx):
    """전역 context 효과 = (그 context 평균 - 전체 평균). fit(train)에서만 계산."""
    d = pd.DataFrame({"_c": np.asarray(ctx), TARGET: fit_df[TARGET].to_numpy()})
    return (d.groupby("_c")[TARGET].mean() - float(fit_df[TARGET].mean())).to_dict()


def transform_cond(df, table, offset, entity_col, ctx, prior_rate, seasons_range, k, name):
    ctx = np.asarray(ctx)
    lookup_idx = pd.MultiIndex.from_arrays([df[entity_col].to_numpy(), ctx,
                                            df["season"].to_numpy() - 1])

    def lk(col):
        p = table.pivot_table(index=["_e", "_c"], columns="season", values=col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        return p.stack(future_stack=True).reindex(lookup_idx).to_numpy()

    s_cell = np.nan_to_num(lk("s").astype(np.float64), nan=0.0)
    n_cell = np.nan_to_num(lk("n").astype(np.float64), nan=0.0)

    off = pd.Series(ctx).map(offset).fillna(0.0).to_numpy(np.float64)
    prior = np.clip(np.asarray(prior_rate, dtype=np.float64) + off, 1e-6, 1 - 1e-6)
    rate_smooth = (s_cell + k * prior) / (n_cell + k)

    return pd.DataFrame({f"{name}_diff": rate_smooth - prior,
                         f"{name}_n": np.log1p(n_cell)}, index=df.index)


# ---------- entity별 '직전 시즌 끝' marginal 성공률 ----------

def build_entity_season_end(df, entity_col, n_col, rate_col):
    """asof_* 컬럼으로 (entity, season) 시즌종료 시점 누적을 복원 (inseason.py와 동일 보정).

    asof_*_n은 '이 투구 직전까지'라 시즌 마지막 행 기준 그 투구 자체가 빠진다(off-by-one).
    마지막 행의 실제 결과를 더해 보정한다."""
    sub = df.sort_values([entity_col, "row_num"])
    last = sub.groupby([entity_col, "season"], as_index=False).last()
    n_before = last[n_col].fillna(0).to_numpy(np.float64)
    outcome = last[TARGET].to_numpy(np.float64)
    N_end = n_before + 1
    S_end = np.round(last[rate_col].fillna(0).to_numpy(np.float64) * n_before) + outcome
    last["prior_success_rate"] = S_end / np.where(N_end == 0, np.nan, N_end)
    return last[[entity_col, "season", "prior_success_rate"]]


def lookup_prior_rate(df, season_end_table, entity_col, global_rate, seasons_range):
    p = season_end_table.pivot(index=entity_col, columns="season", values="prior_success_rate")
    p = p.reindex(columns=seasons_range).ffill(axis=1)
    idx = pd.MultiIndex.from_arrays([df[entity_col].to_numpy(), df["season"].to_numpy() - 1])
    vals = p.stack(future_stack=True).reindex(idx).to_numpy()
    return pd.Series(vals).fillna(global_rate).to_numpy(np.float64)
