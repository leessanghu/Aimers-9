"""script.py에 ingame(경기내 컨디션) aux head 멤버 배선. head0(direct)만 사용."""
p = '../submit/script.py'
src = open(p, encoding='utf-8').read()

old = """    # v75: pitcher/batter 임베딩 MLP. torch 학습, numpy 순수 행렬곱으로 추론(의존성 없음).
    mlp_weights = artifacts.get("mlp_weights")
    mlp_weight = artifacts.get("mlp_weight", 0.0)"""
new = """    # v75: pitcher/batter 임베딩 MLP. torch 학습, numpy 순수 행렬곱으로 추론(의존성 없음).
    mlp_weights = artifacts.get("mlp_weights")
    mlp_weight = artifacts.get("mlp_weight", 0.0)
    # v79: 경기내 컨디션 aux head (head0=y, head1=현재경기 직전까지 누적성공률).
    # 경기간 컨디션 자기상관은 0.047로 죽었지만 경기내는 0.191로 살아있다는 진단에서 도출.
    # train에서만 타깃을 만들고 test는 각 행 자기 피처로 추정 -> Rule 4 준수. head0만 사용.
    ingame_model = artifacts.get("ingame_model")
    ingame_weight = artifacts.get("ingame_weight", 0.0)"""
assert old in src, "anchor1"
src = src.replace(old, new)

old = """                                - pitcherresid_weight - dangerball_weight - mc5_weight
                                - mlp_weight)"""
new = """                                - pitcherresid_weight - dangerball_weight - mc5_weight
                                - mlp_weight - ingame_weight)"""
assert old in src, "anchor2"
src = src.replace(old, new)

old = """        if mlp_weights is not None and mlp_weight > 0:"""
new = """        if ingame_model is not None and ingame_weight > 0:
            heads_ing = np.clip(ingame_model.predict(X), 0.0, 1.0)
            p_ingame = heads_ing[:, 0]
        else:
            p_ingame = None
        if mlp_weights is not None and mlp_weight > 0:"""
assert old in src, "anchor3"
src = src.replace(old, new)

old = """        if p_mlp is not None:
            preds = preds + mlp_weight * p_mlp"""
new = """        if p_mlp is not None:
            preds = preds + mlp_weight * p_mlp
        if p_ingame is not None:
            preds = preds + ingame_weight * p_ingame"""
assert old in src, "anchor4"
src = src.replace(old, new)

old = """                                   p_pitcherresid, p_dangerball, p_mc5, p_mlp)):"""
new = """                                   p_pitcherresid, p_dangerball, p_mc5, p_mlp, p_ingame)):"""
assert old in src, "anchor5"
src = src.replace(old, new)

old = """         f"{' + MLP(w=%.2f)' % mlp_weight if mlp_weight else ''}\""""
new = """         f"{' + MLP(w=%.2f)' % mlp_weight if mlp_weight else ''}"
         f"{' + InGame(w=%.2f)' % ingame_weight if ingame_weight else ''}\""""
assert old in src, "anchor6"
src = src.replace(old, new)

open(p, 'w', encoding='utf-8').write(src)
print("script.py ingame 배선 완료")
