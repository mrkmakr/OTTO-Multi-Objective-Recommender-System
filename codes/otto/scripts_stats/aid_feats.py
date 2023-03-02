import pandas as pd
import os
import polars as pl
import sys

class CFG:
    train_path = "../inputs/train_valid/train_parquet"
    val_path = "../inputs/train_valid/test_parquet"
    val_label_path = "../inputs/train_valid/test_labels.parquet"
    
    aid_feat_path = "../inputs/aid_feats/"

version = sys.argv[1]

if version == "1":
    df = pd.read_parquet(CFG.val_path)
    suffix = ""
elif version == "2":
    df = (pd.concat([
        pd.read_parquet(CFG.train_path), 
        pd.read_parquet(CFG.val_path)], axis = 0))
    df = df[df["ts"] > df["ts"].max() - 3600 * 1000 * 24 * 14].reset_index(drop = True)
    suffix = "2"
elif version == "3":
    df = (pd.concat([
        pd.read_parquet(CFG.train_path), 
        pd.read_parquet(CFG.val_path)], axis = 0))
    df = df[df["ts"] > df["ts"].max() - 3600 * 1000 * 24 * 28].reset_index(drop = True)
    suffix = "3"


for l in ["clicks", "carts", "orders"]:
    df[f"is_{l}"] = df["type"] == l
df = pl.from_pandas(df)

# count infomation
agg = df.groupby("aid").agg([
    pl.count().alias("a_count" + suffix),
    pl.col("session").n_unique().alias("a_session_nunique" + suffix),
    pl.col("is_clicks").sum().alias("a_num_clicks" + suffix),
    pl.col("is_carts").sum().alias("a_num_carts" + suffix),
    pl.col("is_orders").sum().alias("a_num_orders" + suffix),
])

# rate information
agg = agg.with_columns(
    [
        (pl.col(f"a_num_{l}" + suffix) / pl.col("a_count" + suffix) * 10000).cast(pl.UInt32).alias(f"a_rate_{l}" + suffix)
        for l in ["clicks", "carts", "orders"]       
    ]
)
for c in agg.columns:
    agg = agg.with_columns(pl.col(c).cast(pl.UInt32))

os.system("mkdir ../inputs/stats/")
agg.write_parquet(f"../inputs/stats/aid_feats{suffix}.parquet")