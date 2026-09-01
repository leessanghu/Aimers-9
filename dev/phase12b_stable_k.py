"""재현되는 진짜 분산으로부터 축소강도 K를 직접 유도.

핵심: dev_prior와 dev_next는 '같은 안정적 편차'의 서로 독립인 노이즈 관측이다.
따라서 cov(dev_prior, dev_next) = Var(안정적 편차)  (노이즈는 공분산에서 상쇄).
Var(관측)에서 노이즈를 빼는 기존 방식보다 교란(시즌 공존 등)에 훨씬 강건하다.

경험적 베이즈 최적 축소: K = p(1-p) / Var(안정적 편차)
"""
import sys

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

MIN_PRIOR_N, MIN_NEXT_N = 100, 50


def stable_var(df, main_col, ctx, label, base_var):
    d = df.assign(_m=main_col, _c=ctx)
    seasons = sorted(d["season"].unique())
    cell = d.groupby(["_m", "_c", "season"])["control_success"].agg(s="sum", n="count")
    marg = d.groupby(["_m", "season"])["control_success"].agg(s="sum", n="count")

    DP, DN = [], []
    for i in range(1, len(seasons)):
        S, Snext = seasons[i - 1], seasons[i]
        prior = [x for x in seasons if x <= S]
        cp = cell[cell.index.get_level_values("season").isin(prior)].groupby(level=[0, 1]).sum()
        mp = marg[marg.index.get_level_values("season").isin(prior)].groupby(level=0).sum()
        cn = cell[cell.index.get_level_values("season") == Snext].droplevel("season")
        mn = marg[marg.index.get_level_values("season") == Snext].droplevel("season")

        j = cp.join(cn, how="inner", lsuffix="_p", rsuffix="_n")
        j = j[(j["n_p"] >= MIN_PRIOR_N) & (j["n_n"] >= MIN_NEXT_N)]
        if not len(j):
            continue
        keys = j.index.get_level_values(0)
        mps, mpn = mp["s"].reindex(keys).to_numpy(), mp["n"].reindex(keys).to_numpy()
        mns, mnn = mn["s"].reindex(keys).to_numpy(), mn["n"].reindex(keys).to_numpy()
        ok = (mpn > 0) & (mnn > 0) & ~np.isnan(mps) & ~np.isnan(mns)
        if ok.sum() < 30:
            continue
        DP.append((j["s_p"].to_numpy() / j["n_p"].to_numpy())[ok] - (mps / mpn)[ok])
        DN.append((j["s_n"].to_numpy() / j["n_n"].to_numpy())[ok] - (mns / mnn)[ok])

    dp, dn = np.concatenate(DP), np.concatenate(DN)
    cov = float(np.cov(dp, dn)[0, 1])          # = Var(안정적 편차)
    rng = np.random.default_rng(0)
    boots = np.array([np.cov(dp[i], dn[i])[0, 1]
                      for i in (rng.integers(0, len(dp), len(dp)) for _ in range(400))])
    lo, hi = np.percentile(boots, [2.5, 97.5])

    if cov <= 0:
        print(f"{label:26s} cov<=0 -> 사용 불가")
        return None
    sd = np.sqrt(cov)
    k = base_var / cov
    pts = cov / base_var * 100000
    print(f"{label:26s} 안정SD={sd:.5f}  K={k:7.0f}  실질상한~{pts:5.1f}점  "
          f"cov95%CI=[{lo:+.2e}, {hi:+.2e}]{'  (0포함)' if lo <= 0 else ''}")
    return k


def main():
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    g = df.control_success.mean()
    base_var = g * (1 - g)
    print(f"baseline_var = {base_var:.4f}\n")

    print("=== 양성 대조군 (실전 검증됨) ===")
    stable_var(df, df.pitcher_id, df.batter_hand, "투수x타자손 [+17.9]", base_var)
    stable_var(df, df.pitcher_id, np.clip(df.inning, 1, 9), "투수x이닝 [+9.29]", base_var)

    print("\n=== 신규 후보 ===")
    stable_var(df, df.batter_id, df.pitcher_hand, "타자x투수손", base_var)
    stable_var(df, df.pitcher_id, df.balls_before * 4 + df.strikes_before, "투수x볼카운트", base_var)
    stable_var(df, df.pitcher_id, df.game_month, "투수x월", base_var)
    stable_var(df, df.pitcher_id, df.batter_team_id, "투수x상대팀", base_var)


if __name__ == "__main__":
    main()
