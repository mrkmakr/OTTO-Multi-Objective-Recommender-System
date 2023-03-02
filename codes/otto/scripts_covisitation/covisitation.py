# script for calculating covisitation

from tqdm.auto import tqdm
import pandas as pd
import os
import polars as pl
import sys

version = sys.argv[1]

class CFG:
    train_path = "../inputs/train_valid/train_parquet"
    val_path = "../inputs/train_valid/test_parquet"
    val_label_path = "../inputs/train_valid/test_labels.parquet"

## load data
df = pd.concat([
    pd.read_parquet(CFG.train_path), 
    pd.read_parquet(CFG.val_path),
], axis=0).reset_index(drop=True)
print(df.shape)

# use a different parameter according to a version argument
if version == "v12":
    # only recent
    th = df["ts"].max() - 60 * 60 * 24 * 1000 * 14
    df = df[df["ts"] > th].reset_index(drop=True)
    print(df.shape)

# weight depending on type
weights = {"clicks" : 1, "carts" : 3, "orders" : 6}
if version == "v13":
    weights = {"clicks" : 0, "carts" : 3, "orders" : 6}
if version == "v14":
    weights = {"clicks" : 1, "carts" : 9, "orders" : 1}
if version == "v15":
    weights = {"clicks" : 1, "carts" : 0, "orders" : 0}
if version == "v17":
    weights = {"clicks" : 1, "carts" : 9, "orders" : 6}
if version == "v18":
    weights = {"clicks" : 1, "carts" : 15, "orders" : 20}
if version == "v19":
    weights = {"clicks" : 1, "carts" : 9, "orders" : 1}

# how far to consider
n_lookback = 2
if version == "v16":
    n_lookback = 5
if version == "v17":
    n_lookback = 20

# how many candidates to remain
topk = 100
if version == "v19":
    topk = 300
if version == "v20":
    topk = 300

df["weight"] = df["type"].map(weights)
df["chunk"] = df["session"]//10000


def count(df):
    """
    This function counts covisitation.
    It takes a Pandas DataFrame as input, which is expected to have three columns: "session", "aid", and "weight".
    It returns a Pandas DataFrame that contains three columns, "aid_key", "aid_future" and "score".
    """
    ss = df["session"].to_list()
    aa = df["aid"].to_list()
    ww = df["weight"].to_list()
    
    s_ = ss[0]
    a_list = [aa[0]]
    recs1 = []
    for i in (range(1, len(ss))):
        s = ss[i]
        a = aa[i]
        w = ww[i]
        if s_ == s:
            recs1.append([a_list[-n_lookback:], a, w])
        else:
            a_list = []
        a_list = a_list + [a]
        s_ = s

    rec_df = pd.DataFrame(recs1)
    rec_df = pl.from_pandas(rec_df)
    agg = rec_df.explode("0").groupby(["0","1"]).sum()
    agg.columns = ["aid_key", "aid_future", "score"]
    return agg

# Calculate covisitation information
aggs = []
for _, g in tqdm(df.groupby("chunk")):
    if len(g) < 2:
        continue
    agg = count(g)
    aggs.append(agg)
aggs = pl.concat(aggs)
count_df = aggs.groupby(["aid_key", "aid_future"]).sum()

# save calculated results
count_info_list = [-1] * (count_df["aid_key"].max() + 1)
for aid_key, g in tqdm(count_df.groupby("aid_key"), total=len(count_df["aid_key"].unique())):
    # Only save aids with the topk scores
    g = g.sort(by="score", reverse=True)[:topk]
    count_info_list[aid_key] = [g["aid_future"].to_list(), g["score"].to_list()]
os.system("mkdir ../inputs/comatrix")
pd.to_pickle(count_info_list, f"../inputs/comatrix/count_dic_{version}_all.pkl")