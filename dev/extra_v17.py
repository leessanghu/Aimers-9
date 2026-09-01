"""v17에서 쓰는 (C)(D)(E)(F) 피처 — lastyear strike / pitchmix / arsenal JS / pitch_of_pa.

전부 phase43에서 기각선 근처였지만("다 넣어보고 실측으로 검증" 요청에 따라) v17b에 포함한다.
전부 (entity, season) 누적 테이블을 train에서만 만들고 각 행은 직전 시즌까지만 조회한다.
"""

import numpy as np
import pandas as pd

MIX_COLS = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]


def _last_rows(df):
    if "row_num" not in df.columns:
        df = df.assign(row_num=df["row_id"].str.replace("TRAIN_", "", regex=False)
                       .str.replace("TEST_", "", regex=False).astype(int))
    sub = df.sort_values(["pitcher_id", "row_num"])
    return sub.groupby(["pitcher_id", "season"], as_index=False).last()


# ---------- (C) lastyear strike ----------

def build_strike_table(df):
    last = _last_rows(df)
    nb = last["asof_pitcher_n"].fillna(0).to_numpy(np.float64)
    K_end = np.round(last["asof_pitcher_strike_rate"].fillna(0).to_numpy(np.float64) * nb)
    return pd.DataFrame({"pitcher_id": last["pitcher_id"], "season": last["season"],
                         "K_end": K_end, "N_end": nb + 1})


def global_strike_rate(df):
    n = df["asof_pitcher_n"].fillna(0) + 1
    return float(np.average(df["asof_pitcher_strike_rate"].fillna(0), weights=n))


def transform_strike(df, kt, gk, seasons_range, k=30.0):
    piv = {c: kt.pivot(index="pitcher_id", columns="season", values=c)
                .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
           for c in ["K_end", "N_end"]}
    i1 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    i2 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 2])
    v1 = {c: np.nan_to_num(piv[c].reindex(i1).to_numpy().astype(np.float64), nan=0.0) for c in piv}
    v2 = {c: np.nan_to_num(piv[c].reindex(i2).to_numpy().astype(np.float64), nan=0.0) for c in piv}
    n_ly = np.clip(v1["N_end"] - v2["N_end"], 0, None)
    c_ly = np.clip(v1["K_end"] - v2["K_end"], 0, None)
    raw = np.divide(c_ly, n_ly, out=np.full_like(n_ly, np.nan), where=n_ly > 0)
    return pd.DataFrame({"ly_strike": (n_ly * np.nan_to_num(raw, nan=gk) + k * gk) / (n_ly + k)}, index=df.index)


# ---------- (D)(E) lastyear pitchmix + arsenal JS ----------

def build_pitchmix_table(df):
    last = _last_rows(df)
    nmb = last["asof_pitcher_pitchmix_n"].fillna(0).to_numpy(np.float64)
    d = {"pitcher_id": last["pitcher_id"], "season": last["season"], "MN": nmb}
    for c in MIX_COLS:
        d[c] = np.round(last[c].fillna(0).to_numpy(np.float64) * nmb)
    return pd.DataFrame(d)


def global_mix_rates(df):
    n = df["asof_pitcher_pitchmix_n"].fillna(0) + 1
    return {c: float(np.average(df[c].fillna(0), weights=n)) for c in MIX_COLS}


def transform_pitchmix_arsenal(df, mtd, gmix, seasons_range, k=30.0):
    piv = {c: mtd.pivot(index="pitcher_id", columns="season", values=c)
                 .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
           for c in ["MN"] + MIX_COLS}
    i1 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 1])
    i2 = pd.MultiIndex.from_arrays([df["pitcher_id"], df["season"] - 2])
    v1 = {c: np.nan_to_num(piv[c].reindex(i1).to_numpy().astype(np.float64), nan=0.0) for c in piv}
    v2 = {c: np.nan_to_num(piv[c].reindex(i2).to_numpy().astype(np.float64), nan=0.0) for c in piv}
    mn_ly = np.clip(v1["MN"] - v2["MN"], 0, None)

    out = pd.DataFrame(index=df.index)
    ly_p, car_p = [], []
    for c in MIX_COLS:
        cc = np.clip(v1[c] - v2[c], 0, None)
        r_ly = np.divide(cc, mn_ly, out=np.full_like(mn_ly, np.nan), where=mn_ly > 0)
        r_ly = (mn_ly * np.nan_to_num(r_ly, nan=gmix[c]) + k * gmix[c]) / (mn_ly + k)
        r_car = np.divide(v1[c], v1["MN"], out=np.full_like(mn_ly, np.nan), where=v1["MN"] > 0)
        r_car = np.nan_to_num(r_car, nan=gmix[c])
        short = c.split("_")[-2]
        out[f"lymix_{short}"] = r_ly
        out[f"lymix_{short}_minus_career"] = r_ly - r_car
        ly_p.append(r_ly)
        car_p.append(r_car)

    P = np.clip(np.vstack(ly_p).T, 1e-9, None); P /= P.sum(1, keepdims=True)
    Q = np.clip(np.vstack(car_p).T, 1e-9, None); Q /= Q.sum(1, keepdims=True)
    M = 0.5 * (P + Q)
    js = 0.5 * (P * np.log(P / M)).sum(1) + 0.5 * (Q * np.log(Q / M)).sum(1)
    out["arsenal_js"] = np.nan_to_num(js, nan=0.0)
    return out


# ---------- (F) Trackman pitch_of_pa ----------

def build_popa_table(df, tm_path="../data/trackman_history.csv", map_path="pitcher_map.csv"):
    m = pd.read_csv(map_path).sort_values("sim", ascending=False).drop_duplicates("tm_id")
    p2t = m.set_index("pitcher_id")["tm_id"]
    tmh = pd.read_csv(tm_path, encoding="utf-8-sig",
                      usecols=["season", "pitcher_trackman_id", "pitch_of_pa", "balls_before", "strikes_before"])
    tmh = tmh.rename(columns={"pitcher_trackman_id": "tm_id"})
    inv = p2t.reset_index().set_index("tm_id")["pitcher_id"]
    tmh["pitcher_id"] = tmh["tm_id"].map(inv)
    tmh = tmh.dropna(subset=["pitcher_id"])
    tmh["pitcher_id"] = tmh["pitcher_id"].astype(int)
    tmh["count_state"] = tmh["balls_before"] * 4 + tmh["strikes_before"]
    return (tmh.groupby(["pitcher_id", "season", "count_state"])["pitch_of_pa"]
            .agg(popa_mean="mean", popa_max="max").reset_index())


def transform_popa(df, prof, seasons_range):
    cs = (df["balls_before"] * 4 + df["strikes_before"]).to_numpy()
    piv = {c: prof.pivot_table(index=["pitcher_id", "count_state"], columns="season", values=c, aggfunc="first")
               .reindex(columns=seasons_range).ffill(axis=1).stack(future_stack=True)
           for c in ["popa_mean", "popa_max"]}
    idx = pd.MultiIndex.from_arrays([df["pitcher_id"], cs, df["season"] - 1])
    return pd.DataFrame({c: np.nan_to_num(piv[c].reindex(idx).to_numpy().astype(np.float64), nan=0.0)
                         for c in piv}, index=df.index)


def export_stats_cde(kt, gk, mtd, gmix, seasons_range, k_strike=30.0, k_mix=30.0):
    return {"kt": kt, "gk": gk, "mtd": mtd, "gmix": gmix,
            "seasons_range": list(seasons_range), "k_strike": k_strike, "k_mix": k_mix}


def export_stats_f(prof, seasons_range):
    return {"prof": prof, "seasons_range": list(seasons_range)}
