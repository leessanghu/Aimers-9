"""투구 단위 라벨 복원 — 주최측이 안 준 reverse/middle/ball/strike 라벨을 정확히 되살린다.

발견: 같은 투수의 연속된 두 행에서 asof_pitcher_n 증가량이 정확히 +1인 비율이 100.00%다.
      즉 train.csv는 그 투수의 모든 투구를 빠짐없이, 시간순으로 담고 있다.
      따라서 누적 카운트의 차분이 곧 그 투구의 라벨이다:
          label_i = round(rate_{i+1} * n_{i+1}) - round(rate_i * n_i)  in {0,1}

검증: success 축으로 대조한 결과 실제 control_success와 100.000% 일치.
      reverse/middle/ball/strike 4축 모두 0/1 비율 100.00%.

의미: 라벨이 1개(control_success)에서 5개로 늘었다. 조건부 분할을 '성공률'이 아니라
      '실패 유형별'로 만들 수 있고, 4축은 같은 잠재 제구력의 독립적 관측이라
      셀당 유효 정보량이 크게 는다 (조건부 분할 6전 실패의 원인이 셀 노이즈였음).

leakage: train에서만 라벨을 복원해 (entity, 조건, season) 누적 테이블을 만들고,
         각 행은 직전 시즌까지만 조회한다. test 행 간 참조 없음(test는 복원 자체를 안 함).
"""

import numpy as np
import pandas as pd

RATE_COLS = {
    "success": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}
LABELS = ["reverse", "middle", "ball", "strike"]   # success는 이미 주어짐


def recover_pitch_labels(df):
    """각 투구의 reverse/middle/ball/strike 이진 라벨을 복원해 df 인덱스로 반환.

    마지막 투구는 다음 행이 없어 복원 불가 -> NaN (투수당 1행, 전체의 0.04%)."""
    if "row_num" not in df.columns:
        df = df.assign(row_num=df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int))
    d = df.sort_values(["pitcher_id", "row_num"])
    n = d["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    same = np.empty(len(d), dtype=bool)
    same[:-1] = d["pitcher_id"].to_numpy()[1:] == d["pitcher_id"].to_numpy()[:-1]
    same[-1] = False

    out = pd.DataFrame(index=d.index)
    for name in LABELS:
        c = np.round(d[RATE_COLS[name]].fillna(0).to_numpy(np.float64) * n)
        lab = np.full(len(d), np.nan)
        lab[:-1] = c[1:] - c[:-1]
        lab[~same] = np.nan
        out[f"lab_{name}"] = lab
    return out.reindex(df.index)


def build_cond_table(df, labels, ctx, entity="pitcher_id"):
    """(entity, ctx, season) -> 4개 라벨의 시즌 누적 합/개수. train에서만 호출."""
    d = pd.DataFrame({"_e": df[entity].to_numpy(), "_c": np.asarray(ctx),
                      "season": df["season"].to_numpy()}, index=df.index)
    for name in LABELS:
        d[name] = labels[f"lab_{name}"].to_numpy()
    d = d.dropna(subset=LABELS)
    agg = {name: "sum" for name in LABELS}
    agg["_e"] = "size"
    g = d.groupby(["_e", "_c", "season"]).agg(**{name: (name, "sum") for name in LABELS},
                                              n=("season", "size")).sort_index()
    return g.groupby(level=[0, 1]).cumsum().reset_index()


def build_global_offsets(df, labels, ctx):
    """전역 (ctx -> 각 라벨 발생률) 및 전체 평균. 모델이 이미 아는 주효과를 빼기 위함."""
    d = pd.DataFrame({"_c": np.asarray(ctx)}, index=df.index)
    for name in LABELS:
        d[name] = labels[f"lab_{name}"].to_numpy()
    d = d.dropna(subset=LABELS)
    glob = {name: float(d[name].mean()) for name in LABELS}
    by_ctx = d.groupby("_c")[LABELS].mean()
    return glob, by_ctx


def transform_cond_labels(df, table, glob, by_ctx, ctx, prefix, seasons_range, k=400.0,
                          entity="pitcher_id"):
    """각 행에 '직전 시즌까지 누적된 (entity, ctx) 셀의 라벨 발생률 편차'를 붙인다.

    전역 ctx 효과를 prior에 넣어 빼므로, 모델이 이미 아는 주효과와 중복되지 않는다."""
    ctx = np.asarray(ctx)
    idx = pd.MultiIndex.from_arrays([df[entity].to_numpy(), ctx, df["season"].to_numpy() - 1])

    def lk(col):
        p = table.pivot_table(index=["_e", "_c"], columns="season", values=col, aggfunc="first")
        p = p.reindex(columns=seasons_range).ffill(axis=1)
        return np.nan_to_num(p.stack(future_stack=True).reindex(idx).to_numpy().astype(np.float64), nan=0.0)

    n_cell = lk("n")
    out = pd.DataFrame(index=df.index)
    for name in LABELS:
        s_cell = lk(name)
        anchor = pd.Series(ctx).map(by_ctx[name]).fillna(glob[name]).to_numpy(np.float64)
        rate = (s_cell + k * anchor) / (n_cell + k)
        out[f"{prefix}_{name}_dev"] = rate - anchor
    out[f"{prefix}_n"] = np.log1p(n_cell)
    return out


def export_stats(table, glob, by_ctx, seasons_range, k=400.0):
    return {"table": table, "glob": glob, "by_ctx": by_ctx,
            "seasons_range": list(seasons_range), "k": float(k)}
