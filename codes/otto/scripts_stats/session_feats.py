from tqdm.auto import tqdm
import pandas as pd
import numpy as np

class CFG:
    train_path = "../inputs/train_valid/train_parquet"
    val_path = "../inputs/train_valid/test_parquet"
    val_label_path = "../inputs/train_valid/test_labels.parquet"
    
    aid_feat_path = "../inputs/aid_feats/"

df = pd.read_parquet(CFG.val_path)

df["hour"] = df["ts"]//3600//1000%24
df["minute"] = df["ts"]//60//1000%60

for l in ["clicks", "carts", "orders"]:
    df[f"is_{l}"] = df["type"] == l

recs = []
for s, g in tqdm(df.groupby("session")):
    dic = {"session" : s}
    dic["u_length"] = len(g)
    dic["u_aid_dup_rate"] = int(g["aid"].nunique()/len(g) * 10000)
    if len(g) == 1:
        dic["u_last_ts_diff"] = 0
    else:
        dic["u_last_ts_diff"] = (g["ts"].iloc[-1] - g["ts"].iloc[-2])//1000 + 1
    dic["u_session_time_length"] = (g["ts"].iloc[-1] - g["ts"].iloc[0])//1000//60
    dic["u_time_density"] = int((dic["u_session_time_length"] / dic["u_length"]) * 10000)
    for l in ["clicks", "carts", "orders"]:
        dic[f"u_{l}"] = sum(g[f"is_{l}"])
        dic[f"u_{l}_rate"] = int(dic[f"u_{l}"] / len(g) * 10000)
    recs.append(dic)

df = pd.DataFrame(recs)
df = df.astype(np.uint32)
df.to_parquet("../inputs/stats/session_feats.parquet")