from tqdm.auto import tqdm
import pandas as pd
import sys
import glob
import os
import glob
import polars as pl

class CFG:
    path_2nd_val = f"../inputs/2nd_stage/"

def preprocess(df, feats):
    df = df.fill_null(0)

    ## select cols
    non_feats = ["aid", "label_carts", "label_clicks", "label_orders", "session"]
    df = df[feats + non_feats]
    
    ## filter candidates by the features for the label type
    condition = False
    for c in feats:
        condition = condition | (pl.col(c) > 0)
    df = df.filter(condition)
    return df

def inference(label_type, run_ids, dirs):
    model_list = []
    # load trained lgbm rankers
    for run_id in run_ids.split(","):
        try:
            model_path = f'../../otto/inputs/models/LGB_{label_type}_{run_id}.pkl'
            model, feats = pd.read_pickle(model_path)
            print("model load done", model_path)        
            model_list.append(model)
        except Exception as e:
            print(str(e))
            print("fail load ", model_path)
    
    # predict
    preds_rec = []
    for d in tqdm(dirs):
        print(label_type, d)
        df_inf = pl.read_parquet(d)
        df_inf = preprocess(df_inf, feats).to_pandas()
        p = 0
        for model in model_list:
            _p = model.predict(df_inf[feats])
            p += _p # ensemble
        df_inf = df_inf[["session", "aid"]]
        df_inf["score"] = p
        preds_rec.append(pl.from_pandas(df_inf))
    pred_df = pl.concat(preds_rec)
    pred_df = pred_df.sort("score")

    # save predicted scores
    fn = f"{save_path}/pred/pred_{label_type}_{data_idx}.parquet"
    pred_df.write_parquet(fn)
    print("save! : ", fn)

dirs = list(glob.glob(CFG.path_2nd_val+"*"))
run_ids = sys.argv[1]
label_type = sys.argv[2]
suffix = label_type
data_idx = int(sys.argv[3])
print("start lgbm ranker inference : model = {}, label = {}, data_idx = {}".format(run_ids, label_type, data_idx))
save_path = run_ids
os.system(f"mkdir {save_path}")
os.system(f"mkdir {save_path}/pred")
os.system(f"mkdir {save_path}/sub")
# The data are divided into four parts. make predictions on the data corresponding to the given arguments.
inference(label_type, run_ids, dirs[data_idx::4]) 
