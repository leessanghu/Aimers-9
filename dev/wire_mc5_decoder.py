"""script.py의 mc5 디코더를 '절편 포함 선형'으로 확장.

기존: p = proba @ mc5_succ           (절편 없는 순수 내적)
확장: p = mc5_intercept + proba @ mc5_succ

mc5_intercept가 없으면 0이라 기존 동작과 완전히 동일(하위호환).

근거: 5-class 시간분할 검증에서 E[y|c] 디코더 418.25 -> 학습 선형 디코더 580.40(+162.14).
E[y|c]는 P(c)가 정확할 때만 최적인데 실제 상호정보량이 1.58%뿐이라, 부정확한 확률에
극단값(0~0.96)을 곱하면 노이즈가 증폭된다. 학습된 가중치는 절편이 baseline을 잡고
class항은 훨씬 보수적인 범위로 조정한다.
"""
p='../submit/script.py'
src=open(p,encoding='utf-8').read()

old = """    mc5_succ = artifacts.get("mc5_succ")"""
new = """    mc5_succ = artifacts.get("mc5_succ")
    # v84: 디코더 절편. 없으면 0 -> 기존 동작과 동일(하위호환).
    mc5_intercept = float(artifacts.get("mc5_intercept", 0.0))"""
assert old in src, "anchor1"
src = src.replace(old, new, 1)

old = """            p_mc5 = np.clip(proba5 @ np.asarray(mc5_succ, dtype=np.float64), 0.0, 1.0)"""
new = """            p_mc5 = np.clip(mc5_intercept + proba5 @ np.asarray(mc5_succ, dtype=np.float64),
                            0.0, 1.0)"""
assert old in src, "anchor2"
src = src.replace(old, new, 1)

open(p,'w',encoding='utf-8').write(src)
print("script.py mc5 디코더 절편 지원 추가 완료")
