from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import polars as pl

class CFG:
    train_path = "../inputs/train_valid/train_parquet"
    val_path = "../inputs/train_valid/test_parquet"
    val_label_path = "../inputs/train_valid/test_labels.parquet"
    
    aid_feat_path = "../inputs/aid_feats/"

def get_feats(df, roll_num, center, suffix):
    pv = df.pivot_table(values = "session", index = "h", columns = ["aid"], aggfunc = "count")
    pv = pv.fillna(0).astype(int)
    roll = pv.rolling(roll_num, min_periods=1, center = center).sum().astype(int)
    v = roll.values
    v = ((v / v.sum(axis=1, keepdims = True)) * 5000000).astype(int)
    pv_feat = pd.DataFrame(v, index = pv.index, columns = pv.columns)
    pv_feat = pv_feat.astype(np.uint32)
    pv_melt = pv_feat.reset_index().melt("h")
    pv_melt = pv_melt[pv_melt["value"] != 0]
    pv_melt = pv_melt.rename(columns = {"value" : f"ah_popular_{roll_num}_{center}_{suffix}"})
    return pl.from_pandas(pv_melt)

df = pd.read_parquet(CFG.val_path)

df["h"] = df["ts"]//3600//1000
i = 0
for suffix in ["all"] + ["clicks", "carts", "orders"]:
    for roll_num in [6, 24, 48]:
        for center in [True, False]:
            print(suffix, roll_num, center)
            if suffix == "all":
                _feat = get_feats(df, roll_num, center, suffix)
            else:
                _feat = get_feats(df[df["type"] == suffix], roll_num, center, suffix)
            if i == 0:
                feat = _feat
            else:
                feat = feat.join(_feat, on = ["h", "aid"], how = "outer").fill_null(0)
            i += 1

feat = feat.with_columns(
    [pl.col(c).cast(pl.UInt32).alias(c) for c in feat.columns]
)
feat.write_parquet("../inputs/stats/h_aid_rate_v2.parquet")