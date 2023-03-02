import pandas as pd
import numpy as np

class CFG:
    train_path = "../inputs/train_valid/train_parquet"
    val_path = "../inputs/train_valid/test_parquet"
    val_label_path = "../inputs/train_valid/test_labels.parquet"
    aid_feat_path = "../inputs/aid_feats/"

def get_feats(df, name):
    print(name)
    pv = df.pivot_table(values = "session", index = "h", columns = ["aid"], aggfunc = "count")
    pv = pv.fillna(0).astype(int)
    v = pv.values
    v = ((v / v.sum(axis=1, keepdims = True)) * 50000).astype(int)
    pv_feat = pd.DataFrame(v, index = pv.index, columns = pv.columns)
    pv_feat = pv_feat.astype(np.uint32)
    pv_melt = pv_feat.reset_index().melt("h")
    pv_melt = pv_melt[pv_melt["value"] != 0]
    pv_melt_future = pv_melt.copy()
    pv_melt_future["h"] += 1
    feats = pv_melt.merge(pv_melt_future, on = ["h", "aid"], how = "outer").fillna(0).astype(np.uint32)
    feats = feats.rename(columns = {"value_x" : "ah_popular_1h_" + name, "value_y" : "ah_popular_-1h_" + name})
    return feats

df = pd.read_parquet(CFG.val_path)
df["h"] = df["ts"]//3600//1000
feat = get_feats(df, "all")
for l in ["clicks", "carts", "orders"]:
    _feat = get_feats(df[df["type"] == l], l)
    feat = feat.merge(_feat, on = ["h", "aid"], how = "outer").fillna(0).astype(np.uint32)
feat.reset_index(drop = True).to_parquet("../inputs/stats/h_aid_rate.parquet")
