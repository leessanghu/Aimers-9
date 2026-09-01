"""script.py에 mlp_weights(순수 numpy 추론) 멤버 배선. torch 의존성 없음."""
p = '../submit/script.py'
src = open(p, encoding='utf-8').read()

old = """    mc5_model = artifacts.get("mc5_model")
    mc5_weight = artifacts.get("mc5_weight", 0.0)
    mc5_succ = artifacts.get("mc5_succ")"""
new = """    mc5_model = artifacts.get("mc5_model")
    mc5_weight = artifacts.get("mc5_weight", 0.0)
    mc5_succ = artifacts.get("mc5_succ")
    # v75: pitcher/batter 임베딩 MLP. torch 학습, numpy 순수 행렬곱으로 추론(의존성 없음).
    mlp_weights = artifacts.get("mlp_weights")
    mlp_weight = artifacts.get("mlp_weight", 0.0)"""
assert old in src, "anchor1"
src = src.replace(old, new)

old = """                                - pitcherresid_weight - dangerball_weight - mc5_weight)"""
new = """                                - pitcherresid_weight - dangerball_weight - mc5_weight
                                - mlp_weight)"""
assert old in src, "anchor2"
src = src.replace(old, new)

old = """        if mc5_model is not None and mc5_weight > 0 and mc5_succ is not None:
            proba5 = mc5_model.predict_proba(X)
            p_mc5 = np.clip(proba5 @ np.asarray(mc5_succ, dtype=np.float64), 0.0, 1.0)
        else:
            p_mc5 = None"""
new = """        if mc5_model is not None and mc5_weight > 0 and mc5_succ is not None:
            proba5 = mc5_model.predict_proba(X)
            p_mc5 = np.clip(proba5 @ np.asarray(mc5_succ, dtype=np.float64), 0.0, 1.0)
        else:
            p_mc5 = None
        if mlp_weights is not None and mlp_weight > 0:
            w = mlp_weights
            pid_arr = test["pitcher_id"].to_numpy()
            bid_arr = test["batter_id"].to_numpy()
            ip_row = np.array([w["pmap"].get(v, 0) for v in pid_arr], dtype=np.int64)
            ib_row = np.array([w["bmap"].get(v, 0) for v in bid_arr], dtype=np.int64)
            Xrow = X.to_numpy(np.float32)
            z = np.clip((Xrow - w["mu"]) / w["sd"], -10, 10)
            h = np.concatenate([z, w["emb_p"][ip_row], w["emb_b"][ib_row]], axis=1)
            h = np.maximum(h @ w["W1"] + w["b1"], 0)
            h = np.maximum(h @ w["W2"] + w["b2"], 0)
            logit = (h @ w["W3"] + w["b3"]).squeeze(1)
            p_mlp = np.clip(1.0 / (1.0 + np.exp(-logit)), 0.0, 1.0)
        else:
            p_mlp = None"""
assert old in src, "anchor3"
src = src.replace(old, new)

old = """        if p_mc5 is not None:
            preds = preds + mc5_weight * p_mc5"""
new = """        if p_mc5 is not None:
            preds = preds + mc5_weight * p_mc5
        if p_mlp is not None:
            preds = preds + mlp_weight * p_mlp"""
assert old in src, "anchor4"
src = src.replace(old, new)

old = """                                   p_pitcherresid, p_dangerball, p_mc5)):"""
new = """                                   p_pitcherresid, p_dangerball, p_mc5, p_mlp)):"""
assert old in src, "anchor5"
src = src.replace(old, new)

old = """         f"{' + MC5(w=%.2f)' % mc5_weight if mc5_weight else ''}\""""
new = """         f"{' + MC5(w=%.2f)' % mc5_weight if mc5_weight else ''}"
         f"{' + MLP(w=%.2f)' % mlp_weight if mlp_weight else ''}\""""
assert old in src, "anchor6"
src = src.replace(old, new)

open(p, 'w', encoding='utf-8').write(src)
print("script.py mlp 배선 완료")
