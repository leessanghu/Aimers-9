"""Weighted feature-usage audit of the exact v60a production artifact.

HGB models use summed positive split gain; CatBoost models use
PredictionValuesChange. Each model is normalized before applying its v60a
mixture coefficient, so the result is a capacity-allocation audit rather than
a causal/permutation importance estimate.
"""
import joblib
import numpy as np
import pandas as pd

a = joblib.load(r"../submit/model/model_artifacts_v60a.pkl")
features = list(a["feature_order"])
block_map = pd.read_csv("idea57_feature_usage_detail.csv").set_index("feature")["block"].to_dict()

def hgb_gain(model):
    out = np.zeros(len(features), float)
    for stage in model._predictors:
        for predictor in stage:
            nodes = predictor.nodes
            split = nodes["is_leaf"] == 0
            idx = nodes["feature_idx"][split].astype(int)
            gain = np.maximum(nodes["gain"][split].astype(float), 0)
            np.add.at(out, idx, gain)
    return out

def cat_usage(model):
    return np.asarray(model.get_feature_importance(type="PredictionValuesChange"), float)

parts = []
def add(member, model, coeff):
    raw = cat_usage(model) if model.__class__.__module__.startswith("catboost") else hgb_gain(model)
    raw = raw / raw.sum() if raw.sum() else raw
    parts.append(pd.DataFrame({"feature": features, "member": member, "coefficient": coeff,
                               "weighted_usage": coeff*raw}))

# base=.24: HGB/Cat each half, then equal average over three seeds.
for m in a["hgbs"]: add("base", m, .24*.5/len(a["hgbs"]))
for m in a["cats"]: add("base", m, .24*.5/len(a["cats"]))
# hurdle=.32: product of two factors; allocate half to each factor for audit.
for m in a["core_fail_models"]: add("hurdle_core", m, .32*.5/len(a["core_fail_models"]))
for m in a["succ_nc_models"]: add("hurdle_success", m, .32*.5/len(a["succ_nc_models"]))
add("multires", a["multires_model"], .08)
for k in (1,2,3): add("ordinal", a[f"ordinal_stage{k}"], .16/3)
add("midother", a["midother_model"], .20)

d = pd.concat(parts, ignore_index=True)
d["block"] = d["feature"].map(block_map).fillna("unmapped")
detail = d.groupby(["feature", "block"], as_index=False)["weighted_usage"].sum()
detail = detail.sort_values("weighted_usage", ascending=False)
blocks = detail.groupby("block", as_index=False)["weighted_usage"].sum().sort_values("weighted_usage", ascending=False)
by_member = d.groupby(["member", "block"], as_index=False)["weighted_usage"].sum()
by_member["within_member_share"] = by_member["weighted_usage"] / by_member.groupby("member")["weighted_usage"].transform("sum")

detail.to_csv("idea60_v60a_feature_usage_detail.csv", index=False)
blocks.to_csv("idea60_v60a_feature_usage_blocks.csv", index=False)
by_member.to_csv("idea60_v60a_feature_usage_by_member.csv", index=False)
print("v60a weighted model-usage blocks")
print(blocks.to_string(index=False, formatters={"weighted_usage": "{:.4%}".format}))
print("\ntop 20 features")
print(detail.head(20).to_string(index=False, formatters={"weighted_usage": "{:.4%}".format}))
