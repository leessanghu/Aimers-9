"""구종 레퍼토리 엔트로피 — 이 투수의 구종 믹스(fastball/breaking/offspeed)가 얼마나
'예측 불가능'한지를 하나의 스칼라로 압축.

이미 공식 제공되는 asof_pitcher_{fastball,breaking,offspeed}_rate + asof_pitcher_pitchmix_n을
그대로 쓴다(새 정보원이 아니라, 트리가 스스로 잘 못 만드는 비선형 결합을 명시적으로 준다 —
crosses.py와 같은 철학). 그래서 leakage 걱정이 전혀 없다: 그 행 자신의 공식 컬럼만 쓴다.

가설: 레퍼토리가 다양/예측불가능한(엔트로피 높은) 투수는 타자가 다음 구종을 예측하기 어려워
디셉션 효과로 제구(존/유인구 의도 실행)에 유리할 수 있다. 반대로 한 구종에 쏠린(엔트로피 낮은)
투수는 패턴이 읽혀서 불리할 수도 있다.
"""

import numpy as np
import pandas as pd

ARSENAL_COLS = ["arsenal_entropy", "arsenal_top_share"]
K_ARSENAL = 80.0  # asof_pitcher_pitchmix_n 기반 축소 강도 (pitchtype.py의 K_MIX와 동급)
_EPS = 1e-9

_MIX_COLS = ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]


def transform_arsenal(df, global_mix=None, k=K_ARSENAL):
    """global_mix: {col: 전역평균} fit(=train)에서 계산해 고정. None이면 이 df에서 계산(오직 fit 시에만 허용)."""
    n = df["asof_pitcher_pitchmix_n"].fillna(0).to_numpy(np.float64)
    raws = {c: df[c].fillna(0).to_numpy(np.float64) for c in _MIX_COLS}

    if global_mix is None:
        global_mix = {c: float(df[c].mean(skipna=True)) for c in _MIX_COLS}

    shrunk = np.column_stack([
        (n * raws[c] + k * global_mix[c]) / (n + k) for c in _MIX_COLS
    ])
    shrunk = np.clip(shrunk, _EPS, None)
    shrunk = shrunk / shrunk.sum(axis=1, keepdims=True)

    entropy = -(shrunk * np.log(shrunk)).sum(axis=1)
    top_share = shrunk.max(axis=1)

    out = pd.DataFrame(index=df.index)
    out["arsenal_entropy"] = entropy
    out["arsenal_top_share"] = top_share
    return out


def export_stats(global_mix):
    return {"global_mix": global_mix, "k": float(K_ARSENAL)}
