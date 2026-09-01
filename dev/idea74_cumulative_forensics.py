"""누적 피처 포렌식 감사.

목적
1) 모든 asof 누적률을 정수 카운트로 되돌려 숨은 현재 투구 라벨을 감사한다.
2) 투수/타자 양쪽 차분이 같은 라벨을 복원하는지 확인한다.
3) prev1/3/5 game 비율의 숨은 분모를 실제 등판 길이와 대조한다.
4) 경기상태의 다음 행 차분으로 PA 진행/종료 라벨의 오라클 정보량을 잰다.

주의: 다음 행은 train 보조라벨 복원에만 사용한다. test 행 간 참조는 금지다.
이 파일은 연구/검증 전용이며 패키징 코드가 아니다.
"""

import sys
import time

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
T0 = time.time()


def log(msg):
    print(f"[{time.time() - T0:6.1f}s] {msg}", flush=True)


def recover_by_entity(df, entity, n_col, rate_cols):
    """entity별 다음 관측 누적카운트 차분으로 현재 행의 0/1 라벨을 복원."""
    order = np.lexsort((df["row_num"].to_numpy(), df[entity].to_numpy()))
    ent = df[entity].to_numpy()[order]
    n = df[n_col].fillna(0).to_numpy(np.float64)[order]
    same_step = np.r_[((ent[1:] == ent[:-1]) & (np.diff(n) == 1)), False]
    out = {}
    for name, col in rate_cols.items():
        total = np.rint(df[col].fillna(0).to_numpy(np.float64) * df[n_col].fillna(0).to_numpy(np.float64))[order]
        z = np.full(len(df), np.nan)
        delta = np.r_[np.diff(total), np.nan]
        z[order[same_step]] = delta[same_step]
        out[name] = z
    return pd.DataFrame(out), same_step.mean()


def resolution(y, code, valid=None):
    if valid is None:
        valid = np.ones(len(y), dtype=bool)
    d = pd.DataFrame({"code": np.asarray(code)[valid], "y": np.asarray(y)[valid]})
    t = d.groupby("code")["y"].agg(["size", "mean"])
    r = d["y"].mean()
    return float(np.sum(t["size"] / len(d) * (t["mean"] - r) ** 2)), len(t)


def denominator_candidates(rate_a, rate_b, max_n, tol=5.1e-7):
    """각 rate 쌍에 대해 가능한 모든 정수 분모 mask를 반환."""
    q = np.arange(1, max_n + 1, dtype=np.float64)
    a = np.asarray(rate_a, dtype=np.float64)[:, None]
    b = np.asarray(rate_b, dtype=np.float64)[:, None]
    err = np.maximum(np.abs(a * q - np.rint(a * q)), np.abs(b * q - np.rint(b * q))) / q
    return err <= tol


def audit_prev_game_denominators(df, middle_label):
    sig_cols = [f"asof_pitcher_prev{k}_game_{z}_rate" for k in (1, 3, 5) for z in ("success", "middle")]
    d = df[["pitcher_id", "row_num", "season", "control_success", *sig_cols]].copy()
    d["middle"] = middle_label
    d = d.sort_values(["pitcher_id", "row_num"]).reset_index(drop=True)

    sig = d[sig_cols].fillna(-9.0).round(9)
    same_pitcher = d["pitcher_id"].eq(d["pitcher_id"].shift())
    same_sig = sig.eq(sig.shift()).all(axis=1)
    d["outing"] = (~(same_pitcher & same_sig)).cumsum()
    g = d.groupby("outing", sort=False).agg(
        pitcher_id=("pitcher_id", "first"), season=("season", "first"),
        n=("row_num", "size"), success_sum=("control_success", "sum"),
        middle_sum=("middle", "sum"), **{c: (c, "first") for c in sig_cols}
    )
    gp = g.groupby("pitcher_id", sort=False)
    for lag in range(1, 6):
        g[f"n_lag{lag}"] = gp["n"].shift(lag)
        g[f"s_lag{lag}"] = gp["success_sum"].shift(lag)
        g[f"m_lag{lag}"] = gp["middle_sum"].shift(lag)

    log(f"prev-game signature 기준 등판 {len(g):,}개, 등판길이 median={g.n.median():.0f}, p90={g.n.quantile(.9):.0f}")
    for k in (1, 3, 5):
        actual_n = sum(g[f"n_lag{j}"] for j in range(1, k + 1))
        actual_s = sum(g[f"s_lag{j}"] for j in range(1, k + 1))
        actual_m = sum(g[f"m_lag{j}"] for j in range(1, k + 1))
        rs = g[f"asof_pitcher_prev{k}_game_success_rate"]
        rm = g[f"asof_pitcher_prev{k}_game_middle_rate"]
        valid = actual_n.notna() & rs.notna() & rm.notna()
        match = (np.abs(rs[valid] - actual_s[valid] / actual_n[valid]) <= 5.1e-7) & (
            np.abs(rm[valid] - actual_m[valid] / actual_n[valid]) <= 5.1e-7)
        log(f"prev{k}: 실제 최근{k}등판 집계와 rate 일치 {match.mean()*100:.3f}% ({match.sum():,}/{len(match):,})")

        # 실제 집계와 맞는 행에서만 분모 역산법을 공정하게 비교한다.
        ix = np.where(valid.to_numpy())[0][match.to_numpy()]
        max_n = {1: 200, 3: 500, 5: 800}[k]
        # 메모리 제한을 위해 chunk별 통계.
        exact_min = near_global = near_role = total = 0
        abs_min = abs_global = abs_role = 0.0
        cand_sum = unique = 0
        global_prior = float(g.loc[g.season <= 2023, "n"].median()) * k
        role_median = g[g.season <= 2023].groupby("pitcher_id")["n"].median()
        for start in range(0, len(ix), 5000):
            jj = ix[start:start + 5000]
            ok = denominator_candidates(rs.iloc[jj], rm.iloc[jj], max_n)
            q = np.arange(1, max_n + 1)
            truth = actual_n.iloc[jj].to_numpy(np.float64)
            has = ok.any(axis=1)
            qmin = q[np.argmax(ok, axis=1)]
            dist_global = np.where(ok, np.abs(q[None, :] - global_prior), np.inf)
            qglobal = q[np.argmin(dist_global, axis=1)]
            pri = g.iloc[jj].pitcher_id.map(role_median).fillna(global_prior / k).to_numpy(np.float64) * k
            dist_role = np.where(ok, np.abs(q[None, :] - pri[:, None]), np.inf)
            qrole = q[np.argmin(dist_role, axis=1)]
            total += has.sum()
            exact_min += np.sum(qmin[has] == truth[has])
            near_global += np.sum(qglobal[has] == truth[has])
            near_role += np.sum(qrole[has] == truth[has])
            abs_min += np.abs(qmin[has] - truth[has]).sum()
            abs_global += np.abs(qglobal[has] - truth[has]).sum()
            abs_role += np.abs(qrole[has] - truth[has]).sum()
            nc = ok[has].sum(axis=1)
            cand_sum += nc.sum()
            unique += np.sum(nc == 1)
        log(
            f"  분모후보 평균={cand_sum/max(total,1):.1f}, 유일={unique/max(total,1)*100:.2f}% | "
            f"정확도 min={exact_min/max(total,1)*100:.2f}% global근접={near_global/max(total,1)*100:.2f}% "
            f"pitcher-role근접={near_role/max(total,1)*100:.2f}% | "
            f"MAE {abs_min/max(total,1):.1f}/{abs_global/max(total,1):.1f}/{abs_role/max(total,1):.1f}"
        )


def main():
    use = [
        "row_id", "season", "inning", "top_bottom", "balls_before", "strikes_before",
        "pitcher_id", "batter_id", "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate", "asof_batter_n", "asof_batter_success_rate",
        "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate", "asof_pitcher_prev1_game_middle_rate",
        "asof_pitcher_prev3_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
        "control_success",
    ]
    log("train 로드")
    df = pd.read_csv("../data/train.csv", usecols=use, encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "", regex=False).astype(int)
    y = df["control_success"].to_numpy(np.float64)

    pitcher_rates = {
        "success": "asof_pitcher_success_rate", "reverse": "asof_pitcher_reverse_rate",
        "middle": "asof_pitcher_middle_rate", "ball": "asof_pitcher_ball_rate",
        "strike": "asof_pitcher_strike_rate",
    }
    plab, _ = recover_by_entity(df, "pitcher_id", "asof_pitcher_n", pitcher_rates)
    pvalid = plab.notna().all(axis=1).to_numpy()
    log(f"투수 누적 차분 커버리지={pvalid.mean()*100:.4f}%, success 정답일치={(plab.success[pvalid].to_numpy()==y[pvalid]).mean()*100:.6f}%")
    log(f"ball+strike=1 비율={((plab.ball[pvalid]+plab.strike[pvalid])==1).mean()*100:.6f}%")

    mix_rates = {"fastball": "asof_pitcher_fastball_rate", "breaking": "asof_pitcher_breaking_rate", "offspeed": "asof_pitcher_offspeed_rate"}
    ptype, _ = recover_by_entity(df, "pitcher_id", "asof_pitcher_pitchmix_n", mix_rates)
    tvalid = ptype.notna().all(axis=1).to_numpy()
    log(f"구종 차분 커버리지={tvalid.mean()*100:.4f}%, one-hot합=1 비율={(ptype[tvalid].sum(axis=1)==1).mean()*100:.6f}%")

    batter_rates = {"success": "asof_batter_success_rate", "middle": "asof_batter_middle_rate"}
    blab, _ = recover_by_entity(df, "batter_id", "asof_batter_n", batter_rates)
    bvalid = blab.notna().all(axis=1).to_numpy()
    log(f"타자 누적 차분 커버리지={bvalid.mean()*100:.4f}%, success 정답일치={(blab.success[bvalid].to_numpy()==y[bvalid]).mean()*100:.6f}%")
    both = pvalid & bvalid
    log(f"투수/타자 middle 차분 상호일치={(plab.middle[both].to_numpy()==blab.middle[both].to_numpy()).mean()*100:.6f}%, 양쪽 union커버리지={(pvalid|bvalid).mean()*100:.4f}%")

    bits = (plab.loc[pvalid, ["reverse", "middle", "ball", "strike"]].to_numpy(np.int8) * np.array([1, 2, 4, 8])).sum(axis=1)
    ptcode = np.full(len(df), -1, dtype=np.int8)
    ptcode[tvalid] = np.argmax(ptype.loc[tvalid, ["fastball", "breaking", "offspeed"]].to_numpy(), axis=1)
    r_bits, n_bits = resolution(y[pvalid], bits)
    r_cross, n_cross = resolution(y[pvalid], bits * 3 + ptcode[pvalid])
    log(f"joint RMBS {n_bits}클래스 resolution={r_bits:.9f}; x구종 {n_cross}클래스={r_cross:.9f} (증분={r_cross-r_bits:.9f})")

    # 글로벌 행 순서의 다음 경기상태는 train-only 보조라벨이다.
    d = df.sort_values("row_num").reset_index()
    same_pa = np.r_[
        (d.pitcher_id.to_numpy()[1:] == d.pitcher_id.to_numpy()[:-1])
        & (d.batter_id.to_numpy()[1:] == d.batter_id.to_numpy()[:-1])
        & (d.inning.to_numpy()[1:] == d.inning.to_numpy()[:-1])
        & (d.top_bottom.to_numpy()[1:] == d.top_bottom.to_numpy()[:-1]), False
    ]
    b = d.balls_before.to_numpy(); s = d.strikes_before.to_numpy()
    bn = np.r_[b[1:], -99]; sn = np.r_[s[1:], -99]
    event = np.full(len(d), 4, dtype=np.int8)  # 4=PA 종료/그 외
    event[same_pa & (bn == b + 1) & (sn == s)] = 0
    event[same_pa & (sn == s + 1) & (bn == b)] = 1
    event[same_pa & (bn == b) & (sn == s) & (s == 2)] = 2
    event[same_pa & ~(((bn == b + 1) & (sn == s)) | ((sn == s + 1) & (bn == b)) | ((bn == b) & (sn == s) & (s == 2)))] = 3
    event_orig = np.empty(len(d), dtype=np.int8); event_orig[d["index"].to_numpy()] = event
    valid = pvalid & tvalid
    cls5 = np.where(plab.middle.to_numpy() == 1, 0,
                    np.where(plab.reverse.to_numpy() == 1, 1,
                             np.where(plab.ball.to_numpy() == 1, 2,
                                      np.where(plab.strike.to_numpy() == 1, 3, 4))))
    count = df.balls_before.to_numpy() * 3 + df.strikes_before.to_numpy()
    base_code = cls5 * 3 + ptcode
    r15, _ = resolution(y, base_code, valid)
    rev, _ = resolution(y, base_code * 5 + event_orig, valid)
    rcnt, _ = resolution(y, base_code * 12 + count, valid)
    rall, _ = resolution(y, (base_code * 12 + count) * 5 + event_orig, valid)
    log(f"15class resolution={r15:.9f}; xPAevent={rev:.9f}(+{rev-r15:.9f}); xcount={rcnt:.9f}(+{rcnt-r15:.9f}); x둘다={rall:.9f}(+{rall-r15:.9f})")

    audit_prev_game_denominators(df, plab.middle.to_numpy())
    log("완료")


if __name__ == "__main__":
    main()
