"""투수 x 이닝 조건부 분할 피처 — platoon.py와 동일한 검증된 구조.

왜 새 정보인가:
  주최측 asof_* 는 전부 marginal이다. 모델은 `inning`을 원시 피처로 갖고 있어 '전역 이닝 효과'
  (모든 투수 평균으로 몇 이닝이 어렵나)는 알지만, "이 투수가 6이닝에 유독 무너진다"는
  개인 곡선은 볼 수 없다(pitcher_id는 count 인코딩뿐이라 정체성이 없음).
  노이즈 제거 후 이 상호작용의 진짜 SD = 0.0209 -> 상한 약 174점.
  선발의 피로 곡선/타순 3회차, 불펜의 고정 등판 이닝이 투수마다 다르다는 야구적 사실과 일치.

중요: 전역 이닝 효과를 prior에 포함시켜 빼낸다. 그래야 피처가 '순수 개인 상호작용'이 되고
      모델이 이미 아는 inning 주효과와 중복되지 않는다(상한 측정도 같은 방식으로 했다).

leakage 안전성 (platoon/in-season과 동일):
  각 행은 자기 투수의 '직전 시즌 끝까지' 누적된 (pitcher_id, inning) 셀만 조회한다.
  같은 시즌의 다른 행이나 test.csv의 다른 행은 전혀 참조하지 않는다.
  테이블/오프셋은 fit(=train)에서만 만들고 transform은 조회만 한다.

shrinkage: K = p(1-p)/Var(편차) = 0.2494 / 0.0209^2 ~= 570 (경험적 베이즈 최적)
"""

import numpy as np
import pandas as pd

INNING_COLS = ["inning_diff", "inning_n"]
K_INNING = 570.0
MAX_INNING = 9


def inning_bucket(df):
    """연장은 9로 클리핑 (상한 측정 때와 동일)."""
    return np.clip(df["inning"].to_numpy(np.int64), 1, MAX_INNING)


def build_inning_table(df, target_col="control_success"):
    """(pitcher_id, inning, season) -> 그 시즌 끝까지의 누적 (성공수 s, 투구수 n)."""
    d = df.assign(_inn=inning_bucket(df))
    g = (d.groupby(["pitcher_id", "_inn", "season"])[target_col]
           .agg(s="sum", n="count").sort_index())
    cum = g.groupby(level=[0, 1]).cumsum()
    return cum.reset_index()


def build_inning_offset(fit_df, target_col="control_success"):
    """전역 이닝 효과 = (그 이닝 평균 - 전체 평균). fit(train)에서만 계산."""
    d = fit_df.assign(_inn=inning_bucket(fit_df))
    g = d.groupby("_inn")[target_col].mean()
    return (g - float(fit_df[target_col].mean())).to_dict()


def _lookup(table, value_col, seasons_range, lookup_idx):
    p = table.pivot_table(index=["pitcher_id", "_inn"], columns="season",
                          values=value_col, aggfunc="first")
    p = p.reindex(columns=seasons_range).ffill(axis=1)
    return p.stack(future_stack=True).reindex(lookup_idx).to_numpy()


def transform_inning(df, inning_table, inning_offset, pitcher_prior_rate, seasons_range, k=K_INNING):
    """df에 이닝 파생 2개를 붙여 반환.

    pitcher_prior_rate: 각 행의 '직전 시즌 끝 시점' 투수 marginal 성공률(in-season 모듈 재사용)."""
    inn = inning_bucket(df)
    lookup_idx = pd.MultiIndex.from_arrays([df["pitcher_id"], inn, df["season"] - 1])

    s_cell = np.nan_to_num(_lookup(inning_table, "s", seasons_range, lookup_idx).astype(np.float64), nan=0.0)
    n_cell = np.nan_to_num(_lookup(inning_table, "n", seasons_range, lookup_idx).astype(np.float64), nan=0.0)

    # prior = 그 투수의 직전시즌 실력 + 전역 이닝 효과  -> 여기서 벗어난 만큼이 '개인 상호작용'
    off = pd.Series(inn).map(inning_offset).fillna(0.0).to_numpy(np.float64)
    prior = np.clip(np.asarray(pitcher_prior_rate, dtype=np.float64) + off, 1e-6, 1 - 1e-6)

    rate_smooth = (s_cell + k * prior) / (n_cell + k)

    out = pd.DataFrame(index=df.index)
    out["inning_diff"] = rate_smooth - prior
    out["inning_n"] = np.log1p(n_cell)
    return out


def export_stats(inning_table, inning_offset, seasons_range, k=K_INNING):
    return {"inning_table": inning_table, "inning_offset": {int(a): float(b) for a, b in inning_offset.items()},
            "seasons_range": list(seasons_range), "k": float(k)}
