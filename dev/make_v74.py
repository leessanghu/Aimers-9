"""train_final_v62_condball.py를 5-class softmax(v74)용으로 변환."""
src = open('train_final_v62_condball.py', encoding='utf-8').read()

src = src.replace('from catboost import CatBoostRegressor', 'from catboost import CatBoostClassifier')
src = src.replace('CONDBALL_WEIGHT = 0.10', 'MC_WEIGHT = 0.05')
src = src.replace('v50 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v50.pkl"))',
                  'v66 = joblib.load(os.path.join(OUT_DIR, "model_artifacts_v66.pkl"))')
src = src.replace('log(f"  hgbs={len(v50[\'hgbs\'])} cats={len(v50[\'cats\'])} midaxis_weight={v50.get(\'midaxis_weight\')}")',
                  'log(f"  hgbs={len(v66[\'hgbs\'])} cats={len(v66[\'cats\'])}")')
src = src.replace('X = X[v50["feature_order"]].astype(np.float64)', 'X = X[v66["feature_order"]].astype(np.float64)')

old_head = '''lab_reverse = diff_label("asof_pitcher_reverse_rate")
lab_middle = diff_label("asof_pitcher_middle_rate")
lab_ball = diff_label("asof_pitcher_ball_rate")
valid_lab = ~(np.isnan(lab_reverse) | np.isnan(lab_middle) | np.isnan(lab_ball))
dang = valid_lab & ((lab_middle > 0) | (lab_reverse > 0))
notdang = valid_lab & ~dang
log(f"  ラ벨 유효행 {valid_lab.sum():,}/{len(df):,}  not-dangerous {notdang.sum():,}행 ({notdang.mean()*100:.1f}%)")

head_condball = np.where(notdang, 1.0 - lab_ball, np.nan)
Ymat = np.column_stack([y.astype(np.float64), head_condball])'''
# 위 라벨 텍스트가 정확히 안 맞을 수 있으므로 앵커를 나눠서 처리
i0 = src.index('lab_reverse = diff_label')
i1 = src.index('log("cond_ball축 공유트리')
new_head = '''lab_reverse = diff_label("asof_pitcher_reverse_rate")
lab_middle = diff_label("asof_pitcher_middle_rate")
lab_ball = diff_label("asof_pitcher_ball_rate")
lab_strike = diff_label("asof_pitcher_strike_rate")
valid_lab = ~(np.isnan(lab_reverse) | np.isnan(lab_middle) | np.isnan(lab_ball) | np.isnan(lab_strike))
cls = np.full(len(df), -1, dtype=np.int64)
cls[valid_lab & (lab_middle > 0.5)] = 0
cls[valid_lab & (lab_reverse > 0.5) & (lab_middle < 0.5)] = 1
nd = valid_lab & (lab_middle < 0.5) & (lab_reverse < 0.5)
cls[nd & (lab_ball > 0.5)] = 2
cls[nd & (lab_ball < 0.5) & (lab_strike > 0.5)] = 3
cls[nd & (lab_ball < 0.5) & (lab_strike < 0.5)] = 4
fit_mask = cls >= 0
succ_by_cls = np.array([y[fit_mask & (cls == c)].mean() for c in range(5)])
log(f"  labels ok {fit_mask.sum():,}/{len(df):,}  E[y|c]={np.round(succ_by_cls,5)}")
for c in range(5):
    log(f"    class{c}: n={(cls==c).sum():,} ({(cls==c).mean()*100:.2f}%) succ={succ_by_cls[c]*100:.3f}%")

'''
src = src[:i0] + new_head + src[i1:]

old_fit_start = src.index('log("cond_ball축 공유트리')
old_fit_end = src.index('common = dict(v50)')
new_fit = '''log("5-class softmax CatBoost training (full data, labeled rows only)...")
fit_idx = np.where(fit_mask)[0]
n_es = int(len(fit_idx) * 0.92)
tr_i, es_i = fit_idx[:n_es], fit_idx[n_es:]
CAT_PARAMS = dict(iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=5.0, verbose=100,
                  random_seed=42, loss_function="MultiClass", classes_count=5,
                  early_stopping_rounds=40)
ts = time.time()
mc_model = CatBoostClassifier(**CAT_PARAMS)
mc_model.fit(X.iloc[tr_i], cls[tr_i], sample_weight=w[tr_i],
             eval_set=(X.iloc[es_i], cls[es_i]))
log(f"  done best_iter={mc_model.best_iteration_} ({time.time()-ts:.0f}s)")
strip_rng(mc_model)

'''
src = src[:old_fit_start] + new_fit + src[old_fit_end:]

old_save_start = src.index('common = dict(v50)')
old_save_end = src.index('joblib.dump(common, out)')
new_save = '''common = dict(v66)
for k in list(common):
    if k.endswith("_weight") or k == "base_weight":
        common[k] = float(common[k]) * (1.0 - MC_WEIGHT)
common["mc5_model"] = mc_model
common["mc5_succ"] = succ_by_cls
common["mc5_weight"] = MC_WEIGHT
s = sum(float(v) for k, v in common.items() if k.endswith("_weight") or k == "base_weight")
assert abs(s - 1.0) < 1e-9, s
log(f"weights: mc5={MC_WEIGHT:.3f} sum={s:.6f}")

out = os.path.join(OUT_DIR, "model_artifacts_v74.pkl")
'''
src = src[:old_save_start] + new_save + src[old_save_end:]

open('train_final_v74_mc5.py', 'w', encoding='utf-8').write(src)
print("train_final_v74_mc5.py written")
