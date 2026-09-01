"""조건부 분할 후보의 '예측 타당성' 검증.

상한(노이즈 제거 분산)만으로는 부족하다. 상한이 커도 그게 '안정적인 개인 특성'이 아니라
그 시즌 한정의 폼/교란이면, 직전 시즌 테이블로 다음 시즌을 예측할 때 아무 값도 못 낸다.
(투수x월 = 시즌내 폼과 엉킴, 투수x상대팀 = 투수와 팀이 특정 시즌에만 공존하는 교란)

그래서 실제 추론 시점에 하는 일을 그대로 재현해서 잰다:
  편차_prior(p,c) = rate(p,c | 시즌<=S) - rate(p | 시즌<=S)
  편차_next(p,c)  = rate(p,c | 시즌 S+1) - rate(p | 시즌 S+1)
  두 편차의 상관 = 이 조건부가 '재현되는 개인 특성'인지의 직접 측정.

이미 실전에서 통한 platoon(+17.9 추정)/inning(+9.29)을 양성 대조군으로 함께 재서 합격선을 잡는다.
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

MIN_PRIOR_N = 100   # 직전 시즌들 누적 셀 최소 표본
MIN_NEXT_N = 50     # 다음 시즌 셀 최소 표본


def stability(df, main_col, ctx, label, min_prior=MIN_PRIOR_N, min_next=MIN_NEXT_N):
    d = df.assign(_m=main_col, _c=ctx)
    seasons = sorted(d["season"].unique())

    # (main, ctx, season) 및 (main, season) 집계
    cell = d.groupby(["_m", "_c", "season"])["control_success"].agg(s="sum", n="count")
    marg = d.groupby(["_m", "season"])["control_success"].agg(s="sum", n="count")

    dev_prior, dev_next, wts = [], [], []
    for i in range(1, len(seasons)):
        S, Snext = seasons[i - 1], seasons[i]
        prior_seasons = [x for x in seasons if x <= S]

        cp = cell[cell.index.get_level_values("season").isin(prior_seasons)] \
            .groupby(level=[0, 1]).sum()
        mp = marg[marg.index.get_level_values("season").isin(prior_seasons)] \
            .groupby(level=0).sum()
        cn = cell[cell.index.get_level_values("season") == Snext].droplevel("season")
        mn = marg[marg.index.get_level_values("season") == Snext].droplevel("season")

        j = cp.join(cn, how="inner", lsuffix="_p", rsuffix="_n")
        j = j[(j["n_p"] >= min_prior) & (j["n_n"] >= min_next)]
        if not len(j):
            continue
        keys = j.index.get_level_values(0)
        mp_s, mp_n = mp["s"].reindex(keys).to_numpy(), mp["n"].reindex(keys).to_numpy()
        mn_s, mn_n = mn["s"].reindex(keys).to_numpy(), mn["n"].reindex(keys).to_numpy()
        ok = (mp_n > 0) & (mn_n > 0) & ~np.isnan(mp_s) & ~np.isnan(mn_s)
        if ok.sum() < 30:
            continue

        dp = (j["s_p"].to_numpy() / j["n_p"].to_numpy())[ok] - (mp_s / mp_n)[ok]
        dn = (j["s_n"].to_numpy() / j["n_n"].to_numpy())[ok] - (mn_s / mn_n)[ok]
        dev_prior.append(dp)
        dev_next.append(dn)
        wts.append(j["n_n"].to_numpy()[ok])

    if not dev_prior:
        print(f"{label:28s}  표본 부족")
        return None
    dp = np.concatenate(dev_prior)
    dn = np.concatenate(dev_next)
    r = np.corrcoef(dp, dn)[0, 1]
    # 부트스트랩 CI
    rng = np.random.default_rng(0)
    boots = [np.corrcoef(dp[i], dn[i])[0, 1]
             for i in (rng.integers(0, len(dp), len(dp)) for _ in range(300))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"{label:28s}  n={len(dp):6,}  재현상관 r={r:+.4f}  95%CI=[{lo:+.4f}, {hi:+.4f}]")
    return r


def main():
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    print("=== 양성 대조군 (실전에서 이미 통한 것) ===", flush=True)
    stability(df, df["pitcher_id"], df["batter_hand"], "투수x타자손 [실제+17.9]")
    stability(df, df["pitcher_id"], np.clip(df["inning"], 1, 9), "투수x이닝 [실제+9.29]")

    print("\n=== 상한이 컸던 신규 후보 (교란 의심) ===", flush=True)
    stability(df, df["pitcher_id"], df["batter_team_id"], "투수x상대팀 [상한375]")
    stability(df, df["pitcher_id"], df["game_month"], "투수x월 [상한346]")

    print("\n=== 중간 후보 ===", flush=True)
    stability(df, df["pitcher_id"], df["base_state"], "투수x주자상황 [상한62]")
    stability(df, df["pitcher_id"], df["balls_before"] * 4 + df["strikes_before"], "투수x볼카운트 [상한55]")
    stability(df, df["pitcher_id"], np.sign(df["score_diff_pitcher_team"]).astype(int), "투수x점수차 [상한51]")
    stability(df, df["batter_id"], df["pitcher_hand"], "타자x투수손 [상한36]")


if __name__ == "__main__":
    main()
