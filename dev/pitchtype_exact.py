"""구종(fastball/breaking/offspeed) 완전 복원 — Trackman 불필요, train.csv만으로 100% 정확.

발견: asof_pitcher_pitchmix_n이 연속 행에서 정확히 +1씩 증가 (100.00%).
      fastball+breaking+offspeed 델타 합 = 1.0 (100.00%, 예외 없는 진짜 파티션).
      -> 매 투구의 구종을 정수 차분으로 완전히 복원 가능 (Trackman 매칭 불필요).

기존 dev/pitchtype.py 대비:
  기존: Trackman 매칭 정밀도 99.5%, 커버리지 61.5%
  신규: 정밀도 100%, 커버리지 100%, 외부 파일 불필요

leakage 안전성: 이 라벨은 '다음 행'을 봐야 복원되므로(row i의 라벨 = 누적(i+1)-누적(i)),
  그 행 자신의 예측에 쓰면 안 된다. pitchlabels.py와 동일하게 train에서만 라벨을 복원해
  (pitcher, ctx, season) 누적 조회 테이블을 만들고, 각 행은 season-1까지만 조회한다.
"""

import numpy as np
import pandas as pd

TYPES = ["fastball", "breaking", "offspeed"]
RATE_COLS = {"fastball": "asof_pitcher_fastball_rate", "breaking": "asof_pitcher_breaking_rate",
             "offspeed": "asof_pitcher_offspeed_rate"}


def recover_pitchtype_labels(df):
    """각 투구의 구종 원-핫 라벨(fastball/breaking/offspeed)을 복원. 마지막 투구는 NaN."""
    if "row_num" not in df.columns:
        df = df.assign(row_num=df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int))
    d = df.sort_values(["pitcher_id", "row_num"])
    n = d["asof_pitcher_pitchmix_n"].fillna(0).to_numpy(np.float64)
    same = np.empty(len(d), dtype=bool)
    same[:-1] = d["pitcher_id"].to_numpy()[1:] == d["pitcher_id"].to_numpy()[:-1]
    same[-1] = False

    out = pd.DataFrame(index=d.index)
    for t in TYPES:
        c = np.round(d[RATE_COLS[t]].fillna(0).to_numpy(np.float64) * n)
        lab = np.full(len(d), np.nan)
        lab[:-1] = c[1:] - c[:-1]
        lab[~same] = np.nan
        out[f"pt_{t}"] = lab
    return out.reindex(df.index)


def build_control_table(df, pt_labels, entity="pitcher_id"):
    """(entity, ptype, season) -> 시즌 누적 (control_success 합, 개수). 구종별 제구력용."""
    d = pd.DataFrame({"_e": df[entity].to_numpy(), "season": df["season"].to_numpy(),
                      "y": df["control_success"].to_numpy()}, index=df.index)
    for t in TYPES:
        d[t] = pt_labels[f"pt_{t}"].to_numpy()
    d = d.dropna(subset=TYPES)

    rows = []
    for t in TYPES:
        sub = d[d[t] == 1]
        g = sub.groupby(["_e", "season"])["y"].agg(s="sum", n="count").reset_index()
        g["ptype"] = t
        rows.append(g)
    cell = pd.concat(rows, ignore_index=True).sort_values(["_e", "ptype", "season"])
    cell[["s", "n"]] = cell.groupby(["_e", "ptype"])[["s", "n"]].cumsum()
    return cell


def build_mix_table(df, pt_labels, ctx, entity="pitcher_id"):
    """(entity, ctx, ptype, season) -> 시즌 누적 개수. P(ptype | entity, ctx) 추정용."""
    d = pd.DataFrame({"_e": df[entity].to_numpy(), "_c": np.asarray(ctx),
                      "season": df["season"].to_numpy()}, index=df.index)
    for t in TYPES:
        d[t] = pt_labels[f"pt_{t}"].to_numpy()
    d = d.dropna(subset=TYPES)
    rows = []
    for t in TYPES:
        sub = d[d[t] == 1]
        g = sub.groupby(["_e", "_c", "season"]).size().rename("n").reset_index()
        g["ptype"] = t
        rows.append(g)
    cell = pd.concat(rows, ignore_index=True).sort_values(["_e", "_c", "ptype", "season"])
    cell["n"] = cell.groupby(["_e", "_c", "ptype"])["n"].cumsum()
    return cell


def global_type_rates(df, pt_labels):
    d = pd.DataFrame({"y": df["control_success"].to_numpy()}, index=df.index)
    for t in TYPES:
        d[t] = pt_labels[f"pt_{t}"].to_numpy()
    d = d.dropna(subset=TYPES)
    return {t: float(d.loc[d[t] == 1, "y"].mean()) for t in TYPES}


def transform_pitchtype_exact(df, ctrl_table, mix_table, global_rates, prior_rate, ctx, seasons_range,
                              entity="pitcher_id", k_ctrl=340.0, k_mix=80.0):
    """구종 주변화 제구 기대치. dev/pitchtype.py의 transform_pitchtype과 동일한 산출물(pt_pred/pt_dev/pt_n).

    P(success|x) = sum_t P(t|entity,ctx) * ctrl(entity,t)  -- 현재 투구의 실제 구종은 쓰지 않음.
    """
    pid = df[entity].to_numpy()
    season = df["season"].to_numpy()
    ctx = np.asarray(ctx)
    prior = np.asarray(prior_rate, dtype=np.float64)
    n_rows = len(df)
    prev = season - 1

    def piv(table, index_cols, value):
        p = table.pivot_table(index=index_cols, columns="season", values=value, aggfunc="first")
        return p.reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)

    gtype = pd.DataFrame([{"ptype": t, "rate": global_rates[t]} for t in TYPES])

    num_pred = np.zeros(n_rows)
    den_mix = np.zeros(n_rows)
    tot_n = np.zeros(n_rows)

    for t in TYPES:
        ct_s = piv(ctrl_table[ctrl_table["ptype"] == t], "_e", "s")
        ct_n = piv(ctrl_table[ctrl_table["ptype"] == t], "_e", "n")
        mx_n = piv(mix_table[mix_table["ptype"] == t], ["_e", "_c"], "n")

        gm_row = mix_table[mix_table["ptype"] == t].groupby("season")["n"].sum()
        gm_all = mix_table.groupby("season")["n"].sum()

        cs_ = np.nan_to_num(ct_s.reindex(pd.MultiIndex.from_arrays([pid, prev])).to_numpy().astype(np.float64), nan=0.0)
        cn_ = np.nan_to_num(ct_n.reindex(pd.MultiIndex.from_arrays([pid, prev])).to_numpy().astype(np.float64), nan=0.0)
        type_rate = np.divide(cs_, cn_, out=np.full(n_rows, float(global_rates[t])), where=cn_ > 0)

        gm_t = piv(mix_table[mix_table["ptype"] == t].groupby(["_c", "season"])["n"].sum().reset_index(),
                   "_c", "n")
        gmx = np.nan_to_num(gm_t.reindex(pd.MultiIndex.from_arrays([ctx, prev])).to_numpy().astype(np.float64), nan=0.0)

        anchor = np.clip(prior + (type_rate - global_rates[t]), 1e-6, 1 - 1e-6)
        pm = np.nan_to_num(mx_n.reindex(pd.MultiIndex.from_arrays([pid, ctx, prev])).to_numpy().astype(np.float64), nan=0.0)
        ctrl_t = (cs_ + k_ctrl * anchor) / (cn_ + k_ctrl)

        num_pred += (pm + k_mix * gmx) * ctrl_t
        den_mix += (pm + k_mix * gmx)
        tot_n += cn_

    pred = np.divide(num_pred, den_mix, out=prior.copy(), where=den_mix > 0)
    out = pd.DataFrame(index=df.index)
    out["pt_pred"] = pred
    out["pt_dev"] = pred - prior
    out["pt_n"] = np.log1p(tot_n)
    return out


def export_stats(ctrl_table, mix_table, global_rates, seasons_range, k_ctrl=340.0, k_mix=80.0):
    return {"ctrl_table": ctrl_table, "mix_table": mix_table, "global_rates": global_rates,
            "seasons_range": list(seasons_range), "k_ctrl": float(k_ctrl), "k_mix": float(k_mix)}
