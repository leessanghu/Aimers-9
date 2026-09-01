"""시퀀스 구성 로직 검증용 스모크 테스트 (노트북에 옮기기 전 로컬 CPU 확인).

- train 구간: 진짜 point-in-time expanding window (같은 투수, row_num 오름차순, 자기 이전만)
- valid 구간: train_fold 마지막 시점 기준으로 '얼려진' 동일 윈도우 (그 시즌 내에서 안 자람 — 실제
  2025 test 행 독립성 규칙과 동일하게 맞춤)
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

SEQ_LEN = 64
PAD, MASK, FAIL, SUCCESS = 0, 1, 2, 3  # prev_outcome vocab


def load():
    df = pd.read_csv("../data/train.csv", encoding="utf-8-sig")
    df["row_num"] = df["row_id"].str.replace("TRAIN_", "").astype(int)
    return df


def build_expanding(df_sorted_by_pitcher):
    """df_sorted_by_pitcher: pitcher_id, row_num 순 정렬된 df. 각 행 t의 과거 구간
    [group_start[t], t) 인덱스 범위를 반환 (numpy 배열, old baseball project 패턴과 동일)."""
    pid = df_sorted_by_pitcher["pitcher_id"].to_numpy()
    boundary = np.empty(len(pid), bool)
    boundary[0] = True
    boundary[1:] = pid[1:] != pid[:-1]
    group_start = np.maximum.accumulate(np.where(boundary, np.arange(len(pid)), 0))
    return group_start


def make_train_sequences(train_fold, seq_len=SEQ_LEN):
    """train_fold(원본 row_num 순) -> pitcher_id로 재정렬해서 expanding window 인덱스 계산 후
    원래 순서로 복원. 반환: 각 행의 (cont, outcome, pad_mask) 텐서용 numpy 배열들."""
    order = train_fold.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
    sub = train_fold.loc[order].reset_index(drop=True)
    group_start = build_expanding(sub)

    n = len(sub)
    balls = sub["balls_before"].to_numpy(np.float32)
    strikes = sub["strikes_before"].to_numpy(np.float32)
    hand_matchup = (sub["pitcher_hand"] * 10 + sub["batter_hand"]).to_numpy(np.float32)
    outcome_raw = sub["control_success"].to_numpy(np.int64)  # 0/1

    hist_outcome = np.zeros((n, seq_len), dtype=np.int64)  # PAD=0 기본
    hist_balls = np.zeros((n, seq_len), dtype=np.float32)
    hist_strikes = np.zeros((n, seq_len), dtype=np.float32)
    hist_hand = np.zeros((n, seq_len), dtype=np.float32)
    pad_mask = np.ones((n, seq_len), dtype=bool)  # True=패딩

    for t in range(n):
        lo = max(int(group_start[t]), t - seq_len)
        k = t - lo
        if k:
            sl = slice(seq_len - k, seq_len)
            hist_outcome[t, sl] = np.where(outcome_raw[lo:t] == 1, SUCCESS, FAIL)
            hist_balls[t, sl] = balls[lo:t]
            hist_strikes[t, sl] = strikes[lo:t]
            hist_hand[t, sl] = hand_matchup[lo:t]
            pad_mask[t, sl] = False

    # 원래 순서(row_num)로 복원하기 위한 역인덱스
    inv = np.argsort(order)
    return {
        "hist_outcome": hist_outcome[inv], "hist_balls": hist_balls[inv],
        "hist_strikes": hist_strikes[inv], "hist_hand": hist_hand[inv],
        "pad_mask": pad_mask[inv], "k_used": (seq_len - pad_mask.sum(1))[inv],
    }


def make_frozen_context(train_fold, seq_len=SEQ_LEN):
    """각 pitcher_id의 train_fold 내 '마지막' seq_len개 투구 = 그 투수의 고정 컨텍스트.
    valid_fold의 모든 행(같은 투수)이 이 동일 컨텍스트를 공유."""
    sub = train_fold.sort_values(["pitcher_id", "row_num"])
    tail = sub.groupby("pitcher_id").tail(seq_len)

    contexts = {}
    for pid, g in tail.groupby("pitcher_id"):
        g = g.sort_values("row_num")
        outcome = np.where(g["control_success"].to_numpy() == 1, SUCCESS, FAIL)
        balls = g["balls_before"].to_numpy(np.float32)
        strikes = g["strikes_before"].to_numpy(np.float32)
        hand = (g["pitcher_hand"] * 10 + g["batter_hand"]).to_numpy(np.float32)
        k = len(g)
        ho = np.zeros(seq_len, np.int64)
        hb = np.zeros(seq_len, np.float32)
        hs = np.zeros(seq_len, np.float32)
        hh = np.zeros(seq_len, np.float32)
        pm = np.ones(seq_len, bool)
        ho[seq_len - k:] = outcome
        hb[seq_len - k:] = balls
        hs[seq_len - k:] = strikes
        hh[seq_len - k:] = hand
        pm[seq_len - k:] = False
        contexts[pid] = (ho, hb, hs, hh, pm, k)
    return contexts


def apply_frozen_context(valid_fold, contexts, seq_len=SEQ_LEN):
    n = len(valid_fold)
    hist_outcome = np.zeros((n, seq_len), np.int64)
    hist_balls = np.zeros((n, seq_len), np.float32)
    hist_strikes = np.zeros((n, seq_len), np.float32)
    hist_hand = np.zeros((n, seq_len), np.float32)
    pad_mask = np.ones((n, seq_len), bool)
    k_used = np.zeros(n, np.int64)
    for i, pid in enumerate(valid_fold["pitcher_id"].to_numpy()):
        if pid in contexts:
            ho, hb, hs, hh, pm, k = contexts[pid]
            hist_outcome[i], hist_balls[i], hist_strikes[i], hist_hand[i], pad_mask[i] = ho, hb, hs, hh, pm
            k_used[i] = k
    return {"hist_outcome": hist_outcome, "hist_balls": hist_balls, "hist_strikes": hist_strikes,
            "hist_hand": hist_hand, "pad_mask": pad_mask, "k_used": k_used}


def main():
    df = load()
    train_fold = df[df["season"] <= 2023].reset_index(drop=True)
    valid_fold = df[df["season"] == 2024].reset_index(drop=True)

    # 스모크: 앞 50000행만
    tf = train_fold.iloc[:50000].reset_index(drop=True)
    vf = valid_fold.iloc[:5000].reset_index(drop=True)

    print("train expanding sequence 구성...")
    tr_seq = make_train_sequences(tf, seq_len=SEQ_LEN)
    print(" k_used 분포:", pd.Series(tr_seq["k_used"]).describe())

    # 검증: 첫 등장 투수는 k_used=0, 그다음 행부터 1,2,3... 증가해야 함
    check_pid = tf["pitcher_id"].iloc[0]
    mask = tf["pitcher_id"].to_numpy() == check_pid
    ks = tr_seq["k_used"][mask]
    print(f"\n검증(투수 {check_pid} 등장 순서대로 k_used, 앞 10개): {ks[:10]}")
    is_monotonic_capped = all(ks[i] == min(i, SEQ_LEN) for i in range(min(10, len(ks))))
    print("기대대로 0,1,2,...(cap={})".format(SEQ_LEN), "-> 검증", "통과" if is_monotonic_capped else "실패!!")

    print("\nfrozen context 구성 (valid용)...")
    ctx = make_frozen_context(tf, seq_len=SEQ_LEN)
    va_seq = apply_frozen_context(vf, ctx, seq_len=SEQ_LEN)
    print(" k_used 분포(valid):", pd.Series(va_seq["k_used"]).describe())
    seen_ratio = (va_seq["k_used"] > 0).mean()
    print(f" pitcher가 train에 존재해 컨텍스트 있는 비율: {seen_ratio:.3f}")

    # 같은 투수의 여러 valid 행이 정확히 동일한 컨텍스트를 공유하는지 확인
    vf_pids = vf["pitcher_id"].to_numpy()
    dup_pid = pd.Series(vf_pids).value_counts()
    dup_pid = dup_pid[dup_pid > 1].index
    if len(dup_pid):
        p = dup_pid[0]
        idxs = np.where(vf_pids == p)[0][:2]
        same = np.array_equal(va_seq["hist_outcome"][idxs[0]], va_seq["hist_outcome"][idxs[1]])
        print(f"\n동일 투수({p}) 서로 다른 valid 행 2개 컨텍스트 동일 여부: {same} (True여야 함 - 시즌 내 성장 없음 확인)")


if __name__ == "__main__":
    main()
