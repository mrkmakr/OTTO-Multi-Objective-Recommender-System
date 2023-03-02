import pandas as pd
import numpy as np
import polars as pl

class CFG:
    train_path = "../inputs/train_valid/train_parquet"
    val_path = "../inputs/train_valid/test_parquet"
    val_label_path = "../inputs/train_valid/test_labels.parquet"
    aid_feat_path = "../inputs/aid_feats/"

df = pl.from_pandas(pd.concat([
    pd.read_parquet(CFG.train_path), 
    pd.read_parquet(CFG.val_path)], axis = 0))

df = df.with_columns((pl.col("ts")//1000//3600).alias("h"))

d = df.groupby("session").agg(pl.col("h").last()).sort("session").to_numpy().astype(np.uint32)

pd.to_pickle(d, "../inputs/stats/session2h.pkl")