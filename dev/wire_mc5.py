"""script.py에 mc5(5-class softmax) 멤버 배선."""
p = '../submit/script.py'
src = open(p, encoding='utf-8').read()

old = """    # v73: dangerous(=middle or reverse) 행에서만 1-ball. cond_ball의 여집합. head0만 사용.
    dangerball_model = artifacts.get("dangerball_model")
    dangerball_weight = artifacts.get("dangerball_weight", 0.0)"""
new = """    # v73: dangerous(=middle or reverse) 행에서만 1-ball. cond_ball의 여집합. head0만 사용.
    dangerball_model = artifacts.get("dangerball_model")
    dangerball_weight = artifacts.get("dangerball_weight", 0.0)
    # v74: 5-class softmax(middle/reverse/nd&ball/nd&strike/nd&기타).
    # P(success) = sum_c P(c) * E[y|c]. E[y|c]는 train에서 추정해 저장된 값.
    mc5_model = artifacts.get("mc5_model")
    mc5_weight = artifacts.get("mc5_weight", 0.0)
    mc5_succ = artifacts.get("mc5_succ")"""
assert old in src, "anchor1"
src = src.replace(old, new)

old = """                                - pitcherresid_weight - dangerball_weight)"""
new = """                                - pitcherresid_weight - dangerball_weight - mc5_weight)"""
assert old in src, "anchor2"
src = src.replace(old, new)

old = """        if dangerball_model is not None and dangerball_weight > 0:
            heads_db = np.clip(dangerball_model.predict(X), 0.0, 1.0)
            p_dangerball = heads_db[:, 0]
        else:
            p_dangerball = None"""
new = """        if dangerball_model is not None and dangerball_weight > 0:
            heads_db = np.clip(dangerball_model.predict(X), 0.0, 1.0)
            p_dangerball = heads_db[:, 0]
        else:
            p_dangerball = None
        if mc5_model is not None and mc5_weight > 0 and mc5_succ is not None:
            proba5 = mc5_model.predict_proba(X)
            p_mc5 = np.clip(proba5 @ np.asarray(mc5_succ, dtype=np.float64), 0.0, 1.0)
        else:
            p_mc5 = None"""
assert old in src, "anchor3"
src = src.replace(old, new)

old = """        if p_dangerball is not None:
            preds = preds + dangerball_weight * p_dangerball"""
new = """        if p_dangerball is not None:
            preds = preds + dangerball_weight * p_dangerball
        if p_mc5 is not None:
            preds = preds + mc5_weight * p_mc5"""
assert old in src, "anchor4"
src = src.replace(old, new)

old = """                                   p_pitcherresid, p_dangerball)):"""
new = """                                   p_pitcherresid, p_dangerball, p_mc5)):"""
assert old in src, "anchor5"
src = src.replace(old, new)

old = """         f"{' + DangerBall(w=%.2f)' % dangerball_weight if dangerball_weight else ''}\""""
new = """         f"{' + DangerBall(w=%.2f)' % dangerball_weight if dangerball_weight else ''}"
         f"{' + MC5(w=%.2f)' % mc5_weight if mc5_weight else ''}\""""
assert old in src, "anchor6"
src = src.replace(old, new)

open(p, 'w', encoding='utf-8').write(src)
print("script.py mc5 배선 완료")
