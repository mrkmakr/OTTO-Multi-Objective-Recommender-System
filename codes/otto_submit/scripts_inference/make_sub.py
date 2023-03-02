import glob
import pandas as pd
import polars as pl
import sys
from tqdm.auto import tqdm
import os

run_id = sys.argv[1]
print("start creating submission file {}".format(run_id))
pred_dirs = glob.glob(run_id + "/pred/*")

def get_lambda(i):
    return lambda s: s[-1-i]

print("load predicted scores and get top20 for submission")
for d in tqdm(pred_dirs):
    label_type = d.split("_")[-2]
    print(d, label_type)
    # load
    pred_df = pl.read_parquet(d)
    # get top20
    agg = pred_df.groupby("session").agg(
        [
            pl.col("aid").apply(get_lambda(i)).alias(f"aid{i}") for i in range(20) # TODO : use tail
        ]
    )
    # format as a submission file
    recs = []
    for x in agg.to_numpy():
        s = x[0]
        a = " ".join(map(str, list(x[1:])))
        recs.append([s, a])
    pred_df = pd.DataFrame(recs, columns = [f"session_type", "labels"])
    pred_df["session_type"] = pred_df["session_type"].astype(str) + "_" + label_type
    d_dst = d.replace("pred", "sub").replace("parquet", "csv")

    # save partial submission file
    pred_df.to_csv(d_dst, index=False)
    
print("concat all partial submission files and create final submission file zip")
dirs = glob.glob(f"./{run_id}/sub/*")
dfs = []
for d in dirs:
    print(d)
    df = pd.read_csv(d)
    dfs.append(df)
df = pd.concat(dfs, axis = 0)
fn = f"./{run_id}/submission_{run_id}.csv"
df.to_csv(fn, index = False)
print("save! : ", fn)
cmd = f"zip {fn}.zip {fn}"
print(cmd)
os.system(cmd)