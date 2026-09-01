"""벡터화(groupby+shift) 시퀀스 구성이 loop 기반 참조 구현과 동일한 결과를 내는지 검증.
전체 데이터(120만+ 행) 규모에선 loop가 너무 느려서(노트북에서 GPU 학습 전에 시퀀스부터
못 만듦) groupby+shift로 바꿔야 한다 — 이게 정확히 같은 결과인지 먼저 확인.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

from seq_smoketest import load, make_train_sequences, SEQ_LEN, PAD, FAIL, SUCCESS


def build_hist_matrix_vectorized(df_sorted, seq_len):
    """df_sorted: pitcher_id, row_num 순 정렬된 df. groupby+shift로 벡터화."""
    g = df_sorted.groupby("pitcher_id", sort=False)
    hand = (df_sorted["pitcher_hand"] * 10 + df_sorted["batter_hand"])
    outcome_cols, balls_cols, strikes_cols, hand_cols = [], [], [], []
    for j in range(seq_len, 0, -1):
        outcome_cols.append(g["control_success"].shift(j))
        balls_cols.append(g["balls_before"].shift(j))
        strikes_cols.append(g["strikes_before"].shift(j))
        hand_cols.append(hand.groupby(df_sorted["pitcher_id"]).shift(j))

    outcome_mat = np.stack([c.to_numpy(dtype=np.float64) for c in outcome_cols], axis=1)
    balls_mat = np.stack([c.to_numpy(dtype=np.float64) for c in balls_cols], axis=1)
    strikes_mat = np.stack([c.to_numpy(dtype=np.float64) for c in strikes_cols], axis=1)
    hand_mat = np.stack([c.to_numpy(dtype=np.float64) for c in hand_cols], axis=1)

    pad_mask = np.isnan(outcome_mat)
    hist_outcome = np.where(pad_mask, PAD, np.where(outcome_mat == 1, SUCCESS, FAIL)).astype(np.int64)
    hist_balls = np.nan_to_num(balls_mat).astype(np.float32)
    hist_strikes = np.nan_to_num(strikes_mat).astype(np.float32)
    hist_hand = np.nan_to_num(hand_mat).astype(np.float32)
    k_used = (~pad_mask).sum(1)
    return {"hist_outcome": hist_outcome, "hist_balls": hist_balls, "hist_strikes": hist_strikes,
            "hist_hand": hist_hand, "pad_mask": pad_mask, "k_used": k_used}


def main():
    df = load()
    train_fold = df[df["season"] <= 2023].reset_index(drop=True)
    tf = train_fold.iloc[:50000].reset_index(drop=True)

    ref = make_train_sequences(tf, seq_len=SEQ_LEN)  # loop 기반(정답으로 검증된 것)

    order = tf.sort_values(["pitcher_id", "row_num"]).index.to_numpy()
    sub = tf.loc[order].reset_index(drop=True)
    vec_sorted = build_hist_matrix_vectorized(sub, SEQ_LEN)
    inv = np.argsort(order)
    vec = {k: v[inv] for k, v in vec_sorted.items()}

    for key in ["hist_outcome", "hist_balls", "hist_strikes", "hist_hand", "k_used"]:
        same = np.array_equal(ref[key], vec[key])
        print(f"{key}: 일치={same}")
        if not same:
            diff = np.abs(ref[key].astype(float) - vec[key].astype(float))
            print("  최대 차이:", diff.max(), " 불일치 개수:", (diff > 1e-6).sum())

    same_pad = np.array_equal(ref["pad_mask"], vec["pad_mask"])
    print(f"pad_mask: 일치={same_pad}")


if __name__ == "__main__":
    main()
