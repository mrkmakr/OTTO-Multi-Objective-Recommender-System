import sys
import numpy as np
import pandas as pd
import glob
import os
import time
import polars as pl
from tqdm.auto import tqdm
from lightgbm import LGBMRanker
from pprint import pprint
time_start = time.time()

class CFG:
    path_2nd = f"../inputs/2nd_stage/"

def get_group(sessions):
    """
    get session groups for lgbm reranker training
    """
    where = np.where(np.diff(sessions) > 0)[0]
    return np.append(
        np.append(where[0] + 1, np.diff(where)), len(sessions) - where[-1] - 1
    )

def train(x, y, sessions, feats, label_type, param, use_all_data_for_train, epochs):
    """
    train lgbm reranker and save model
    """
    
    if use_all_data_for_train:
        tr_idx = sessions > -1
        va_idx = sessions % 5 == 0
    else:
        tr_idx = sessions % 5 != 0
        va_idx = sessions % 5 == 0
        
    print("n_pos_clicks : ", sum(y[:, 0]))
    print("n_pos_carts : ", sum(y[:, 1]))
    print("n_pos_orders : ", sum(y[:, 2]))
    label2idx = {
        "clicks": 0,
        "carts": 1,
        "orders": 2,
    }
    label_idx = label2idx[label_type]
    print(f"***** label_type={label_type}")
    print(sum(tr_idx), sum(va_idx), len(feats))

    model = LGBMRanker(
        n_estimators=epochs[label_type],
        **param
    )

    model.fit(
        x[tr_idx],
        y[tr_idx, label_idx],
        group=get_group(sessions[tr_idx]),
        eval_set=[(x[va_idx], y[va_idx, label_idx])],
        eval_group=[get_group(sessions[va_idx])],
        eval_at=[20, 100],
        verbose=20,
        early_stopping_rounds=50,
    )
    os.system("mkdir ../inputs/models")
    pd.to_pickle([model, feats], f"../inputs/models/LGB_{label_type}_{run_id}.pkl")

    fi = model.booster_.feature_importance("gain")
    importances = {feats[i]: fi[i] for i in range(len(feats))}
    pprint(sorted(importances.items(), key=lambda x: x[1], reverse=True))

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

def get_xy(dirs, feats, label_type, use_neg_rate, is_sample):
    df = load_data(dirs, feats, label_type, use_neg_rate, is_sample)
    print("load data done")
    df = df.sort(["session", "aid"])
    print("data sorted")

    non_feats = ["aid", "label_carts", "label_clicks", "label_orders", "session"]
    feats = sorted([c for c in df.columns if c not in non_feats])
    print("feats : ", feats)
    print("not feats : ", non_feats)

    x = df[feats].to_numpy()
    y = df[["label_clicks", "label_carts", "label_orders"]].fill_null(0).to_numpy()
    s = df["session"].to_numpy()
    return x, y, s, feats, label_type


def filtering(df, label_type):
    """
    reduce candidates when the number of candidates is bigger than 9999
    because lgb group does not support 10000 or more
    """
    
    fil = (
        df.sort(f"label_{label_type}", reverse=True)
        .groupby("session")
        .agg(pl.col("aid").head(9999))
        .explode(pl.col("aid"))
    ) 
    df = df.join(fil, on=["aid", "session"], how="inner")
    return df


def load_data(dirs, feats, label_type, use_neg_rate, is_sample):
    datas = []
    gb = 0
    for i, d in enumerate(tqdm(dirs)):
        df = pl.read_parquet(d)
        df = preprocess(df, feats)

        if i == 0:
            col_names = df.columns
        df = df.select(col_names)

        # get positive sample
        is_label = (pl.col(f"label_{label_type}")).alias("is_label")
        pos = df.filter(is_label > 0)
        df = df.join(pos.select("session").unique(), on="session", how="inner")
        if df.estimated_size("gb") == 0:
            continue
            
        # get negative sample
        if is_sample: # random sampling
            neg = df.filter(is_label == 0).sample(frac = use_neg_rate)
        else: # sample from predetermined indices
            neg = df.filter(is_label == 0)[::int(1/use_neg_rate)]
            
        # store concatted df
        df = pl.concat([pos, neg])
        df = filtering(df, label_type)
        datas.append(df)
        
        # show expected memory usage 
        gb += df.estimated_size("gb")
        if i % 10 == 0:
            print(f"{gb} GB now | {gb * len(dirs) / len(datas)} GB total estimated")
    return pl.concat(datas)

is_full_train = True
if is_full_train: # for training with full data
    ran = range(0,20)
    is_sample = True
    use_neg_rate = 0.33 * 1.3 * 1.3 * 1.2
else: # for small local validation
    ran = range(10, 11)
    is_sample = False
    use_neg_rate = 0.5

dirs = []
for i in ran:
    dirs += list(glob.glob(CFG.path_2nd + "*_{:03}*".format(i)))
# dirs = dirs[:3]

run_id = sys.argv[1]

# features name for each task
feats_clicks = ['a_count2', 'a_count2_rank', 'a_count3', 'a_count3_rank', 'a_num_carts3', 'a_num_carts3_rank', 'a_num_clicks2', 'a_num_clicks2_rank', 'a_num_clicks3', 'a_num_clicks3_rank', 'a_rate_carts2', 'a_rate_carts2_rank', 'a_rate_carts3', 'a_rate_carts3_rank', 'a_rate_carts_rank', 'a_rate_clicks', 'a_rate_clicks2', 'a_rate_clicks2_rank', 'a_rate_clicks3', 'a_rate_clicks3_rank', 'a_rate_clicks_rank', 'a_rate_orders2', 'a_rate_orders3', 'a_rate_orders_rank', 'a_session_nunique2', 'a_session_nunique3', 'a_session_nunique3_rank', 'ah_popular_-1h_orders_rank', 'ah_popular_24_False_orders_rank', 'ah_popular_24_True_all', 'ah_popular_24_True_all_rank', 'ah_popular_24_True_carts_rank', 'ah_popular_24_True_clicks', 'ah_popular_24_True_clicks_rank', 'ah_popular_24_True_orders_rank', 'ah_popular_48_True_all', 'ah_popular_48_True_clicks', 'ah_popular_48_True_clicks_rank', 'ah_popular_6_False_all_rank', 'ah_popular_6_False_carts_rank', 'ah_popular_6_False_clicks', 'ah_popular_6_False_clicks_rank', 'ah_popular_6_False_orders_rank', 'ah_popular_6_True_all', 'ah_popular_6_True_all_rank', 'ah_popular_6_True_carts_rank', 'ah_popular_6_True_clicks', 'ah_popular_6_True_clicks_rank', 'ah_popular_6_True_orders_rank', 'count_dic_v11_p_chain_all_100_2_all_0', 'count_dic_v11_p_chain_full_100_5_1_0', 'count_dic_v11_p_chain_full_100_5_1_1', 'count_dic_v11_p_chain_full_100_5_1_2', 'count_dic_v11_p_chain_full_100_5_1_3', 'count_dic_v11_p_chain_full_100_5_3_0', 'count_dic_v11_p_chain_full_100_5_7_4', 'count_dic_v11_p_chain_full_20_5_1_0', 'count_dic_v11_p_chain_full_20_5_1_1', 'count_dic_v11_p_chain_full_20_5_1_2', 'count_dic_v11_p_chain_full_20_5_1_3', 'count_dic_v11_p_chain_full_20_5_1_4', 'count_dic_v11_p_chain_full_20_5_3_1', 'count_dic_v11_p_chain_full_20_5_3_2', 'count_dic_v11_p_chain_full_20_5_3_3', 'count_dic_v11_p_chain_w_100_2_all_0', 'count_dic_v12_p_chain_all_100_2_all_0', 'count_dic_v12_p_chain_all_100_2_all_1', 'count_dic_v12_p_chain_all_20_2_all_1', 'count_dic_v14_p_chain_all_100_2_all_0', 'count_dic_v14_p_chain_w_20_2_all_1', 'count_dic_v15_14_chain_1_2dic_all_1', 'count_dic_v15_14_chain_2_2dic_all_2', 'count_dic_v15_14_chain_3_2dic_all_3', 'count_dic_v15_p_chain_all_100_2_all_0', 'count_dic_v15_p_chain_all_100_2_all_1', 'count_dic_v15_p_chain_all_20_2_all_0', 'count_dic_v15_p_chain_all_20_2_all_1', 'count_dic_v16_p_chain_all_100_2_all_0', 'count_dic_v16_p_chain_all_20_2_all_1', 'count_dic_v17_p_chain_all_100_2_all_0', 'count_dic_v17_p_chain_all_100_2_all_1', 'count_dic_v17_p_chain_all_20_2_all_1', 'count_dic_v19_p_chain_w_20_2_all_0', 'count_dic_v20_p_chain_w_20_2_all_0', 'count_dic_v20_p_chain_w_20_2_all_1', 'emb_carts_v18', 'emb_carts_v21', 'emb_carts_v23', 'emb_carts_v27', 'emb_carts_v29', 'emb_carts_v31', 'emb_carts_v42', 'emb_clicks_v15', 'emb_clicks_v18', 'emb_clicks_v21', 'emb_clicks_v23', 'emb_clicks_v27', 'emb_clicks_v29', 'emb_clicks_v31', 'emb_clicks_v42', 'emb_orders_v18', 'emb_orders_v27', 'emb_orders_v29', 'emb_orders_v31', 'emb_orders_v42', 'isin_0_carts', 'isin_0_rank', 'isin_0_ts_diff_post', 'isin_0_ts_diff_pre', 'u_aid_dup_rate', 'u_last_ts_diff', 'u_time_density']
feats_carts = ['a_count2', 'a_count2_rank', 'a_count3', 'a_count3_rank', 'a_num_clicks2', 'a_num_clicks3', 'a_num_clicks3_rank', 'a_rate_carts2', 'a_rate_carts3', 'a_rate_clicks2_rank', 'a_rate_clicks3', 'a_rate_clicks3_rank', 'a_rate_clicks_rank', 'a_session_nunique2', 'a_session_nunique3', 'a_session_nunique3_rank', 'ah_popular_24_False_orders_rank', 'ah_popular_24_True_all', 'ah_popular_24_True_carts_rank', 'ah_popular_24_True_orders_rank', 'ah_popular_48_True_all', 'ah_popular_48_True_clicks', 'ah_popular_48_True_clicks_rank', 'ah_popular_6_False_carts_rank', 'ah_popular_6_False_orders_rank', 'ah_popular_6_True_all', 'ah_popular_6_True_all_rank', 'ah_popular_6_True_carts_rank', 'ah_popular_6_True_clicks_rank', 'ah_popular_6_True_orders_rank', 'count_dic_v11_p_chain_full_100_5_1_0', 'count_dic_v11_p_chain_full_100_5_1_1', 'count_dic_v11_p_chain_full_100_5_1_2', 'count_dic_v11_p_chain_full_100_5_3_0', 'count_dic_v11_p_chain_full_20_5_1_0', 'count_dic_v11_p_chain_full_20_5_1_1', 'count_dic_v11_p_chain_full_20_5_1_2', 'count_dic_v11_p_chain_full_20_5_1_3', 'count_dic_v11_p_chain_full_20_5_1_4', 'count_dic_v11_p_chain_full_20_5_3_2', 'count_dic_v12_p_chain_all_100_2_all_0', 'count_dic_v12_p_chain_all_100_2_all_1', 'count_dic_v14_p_chain_all_100_2_all_0', 'count_dic_v15_14_chain_3_2dic_all_3', 'count_dic_v15_p_chain_all_100_2_all_0', 'count_dic_v16_p_chain_all_100_2_all_0', 'count_dic_v17_p_chain_all_100_2_all_0', 'count_dic_v19_p_chain_w_20_2_all_0', 'count_dic_v20_p_chain_w_20_2_all_1', 'emb_carts_v18', 'emb_carts_v21', 'emb_carts_v23', 'emb_carts_v27', 'emb_carts_v29', 'emb_carts_v31', 'emb_carts_v42', 'emb_clicks_v15', 'emb_clicks_v18', 'emb_clicks_v21', 'emb_clicks_v23', 'emb_clicks_v27', 'emb_clicks_v29', 'emb_clicks_v31', 'emb_clicks_v42', 'emb_orders_v18', 'emb_orders_v27', 'emb_orders_v29', 'emb_orders_v31', 'emb_orders_v42', 'isin_0_carts', 'isin_0_rank', 'isin_0_ts_diff_post', 'u_aid_dup_rate', 'u_last_ts_diff', 'a_num_carts_rank', 'a_rate_carts', 'a_session_nunique2_rank', 'ah_popular_1h_carts', 'ah_popular_24_False_carts_rank', 'ah_popular_24_True_carts', 'ah_popular_48_False_orders_rank', 'ah_popular_48_True_carts', 'ah_popular_48_True_carts_rank', 'ah_popular_6_True_carts', 'count_dic_v11_14_chain_2_2dic_all_2', 'count_dic_v11_p_chain_full_100_5_1_4', 'count_dic_v11_p_chain_full_20_5_5_4', 'count_dic_v11_p_chain_full_20_5_9_3', 'count_dic_v13_p_chain_all_100_2_all_0', 'count_dic_v13_p_chain_all_100_2_all_1', 'count_dic_v13_p_chain_all_20_2_all_1', 'count_dic_v14_p_chain_all_20_2_all_1', 'count_dic_v14_p_chain_w_100_2_all_0', 'count_dic_v14_p_chain_w_100_2_all_1', 'count_dic_v18_p_chain_w_100_2_all_0', 'count_dic_v18_p_chain_w_100_2_all_1', 'count_dic_v18_p_chain_w_20_2_all_1', 'count_dic_v19_p_chain_w_20_2_all_1', 'emb_carts_v15', 'emb_orders_v15', 'isin_0_count', 'u_carts_rate', 'u_clicks', 'u_clicks_rate', 'u_length']
feats_orders = ['a_count3', 'a_count3_rank', 'a_num_carts3', 'a_num_carts3_rank', 'a_num_clicks2', 'a_num_clicks2_rank', 'a_num_clicks3', 'a_num_clicks3_rank', 'a_rate_carts2', 'a_rate_carts3', 'a_rate_carts3_rank', 'a_rate_clicks2_rank', 'a_rate_clicks3', 'a_rate_clicks3_rank', 'a_rate_clicks_rank', 'a_rate_orders2', 'a_rate_orders3', 'a_rate_orders_rank', 'a_session_nunique3', 'a_session_nunique3_rank', 'ah_popular_24_True_orders_rank', 'ah_popular_48_True_clicks', 'ah_popular_6_False_carts_rank', 'ah_popular_6_True_all_rank', 'ah_popular_6_True_carts_rank', 'ah_popular_6_True_orders_rank', 'count_dic_v11_p_chain_full_100_5_1_0', 'count_dic_v11_p_chain_full_100_5_1_1', 'count_dic_v11_p_chain_full_100_5_1_2', 'count_dic_v11_p_chain_full_100_5_7_4', 'count_dic_v11_p_chain_full_20_5_1_1', 'count_dic_v11_p_chain_full_20_5_1_2', 'count_dic_v11_p_chain_full_20_5_1_3', 'count_dic_v11_p_chain_full_20_5_3_1', 'count_dic_v11_p_chain_full_20_5_3_3', 'count_dic_v11_p_chain_w_100_2_all_0', 'count_dic_v12_p_chain_all_100_2_all_0', 'count_dic_v12_p_chain_all_100_2_all_1', 'count_dic_v14_p_chain_all_100_2_all_0', 'count_dic_v14_p_chain_w_20_2_all_1', 'count_dic_v15_14_chain_2_2dic_all_2', 'count_dic_v16_p_chain_all_100_2_all_0', 'count_dic_v16_p_chain_all_20_2_all_1', 'count_dic_v17_p_chain_all_100_2_all_0', 'count_dic_v19_p_chain_w_20_2_all_0', 'count_dic_v20_p_chain_w_20_2_all_1', 'emb_carts_v18', 'emb_carts_v23', 'emb_carts_v27', 'emb_carts_v29', 'emb_carts_v31', 'emb_carts_v42', 'emb_clicks_v21', 'emb_clicks_v23', 'emb_clicks_v27', 'emb_clicks_v29', 'emb_clicks_v31', 'emb_clicks_v42', 'emb_orders_v18', 'emb_orders_v27', 'emb_orders_v29', 'emb_orders_v31', 'emb_orders_v42', 'isin_0_carts', 'isin_0_rank', 'isin_0_ts_diff_post', 'isin_0_ts_diff_pre', 'u_aid_dup_rate', 'u_last_ts_diff', 'a_rate_carts', 'a_session_nunique2_rank', 'ah_popular_1h_carts', 'ah_popular_24_False_carts_rank', 'ah_popular_48_True_carts', 'count_dic_v11_14_chain_2_2dic_all_2', 'count_dic_v11_p_chain_full_100_5_1_4', 'count_dic_v11_p_chain_full_20_5_5_4', 'count_dic_v13_p_chain_all_100_2_all_0', 'count_dic_v13_p_chain_all_20_2_all_1', 'count_dic_v14_p_chain_all_20_2_all_1', 'count_dic_v14_p_chain_w_100_2_all_0', 'count_dic_v18_p_chain_w_100_2_all_0', 'count_dic_v18_p_chain_w_100_2_all_1', 'count_dic_v18_p_chain_w_20_2_all_1', 'count_dic_v19_p_chain_w_20_2_all_1', 'emb_carts_v15', 'emb_orders_v15', 'u_carts_rate', 'u_clicks', 'u_clicks_rate', 'u_length', 'a_num_orders2_rank', 'a_num_orders3', 'a_num_orders3_rank', 'a_rate_orders2_rank', 'a_rate_orders3_rank', 'ah_popular_1h_carts_rank', 'ah_popular_24_False_carts', 'ah_popular_24_True_orders', 'ah_popular_48_False_carts', 'ah_popular_48_True_orders', 'ah_popular_48_True_orders_rank', 'ah_popular_6_False_all', 'ah_popular_6_False_carts', 'count_dic_v14_p_chain_w_20_2_all_0', 'emb_orders_v21', 'emb_orders_v23', 'isin_0_orders', 'u_carts', 'u_orders_rate', 'u_session_time_length']
use_neg_rate_clicks = 0.08
use_neg_rate_carts = 0.35
use_neg_rate_orders = 0.55
if run_id == "v1_simple":
    # simple version
    use_all_data_for_train = False
    param = {'lambda_l1': 2.3568477993751684, 'lambda_l2': 1.2824074933740028e-07, 'num_leaves': 17, 'feature_fraction': 0.8114194133521475, 'bagging_fraction': 0.8548900579828358, 'bagging_freq': 4, 'min_child_samples': 26}
    epochs = {"clicks":5000,"carts":5000,"orders":5000}
    # use only 5 feats
    feats_clicks = ["emb_clicks_v42",
               "count_dic_v15_p_chain_all_20_2_all_0",
               "isin_0_rank",
               "count_dic_v15_p_chain_all_20_2_all_1",
               "isin_0_carts"
              ]
    feats_carts = ["emb_carts_v42",
               "count_dic_v19_p_chain_w_20_2_all_0",
               "count_dic_v16_p_chain_all_100_2_all_0",
               "count_dic_v14_p_chain_all_20_2_all_1",
               "isin_0_carts"
              ]
    feats_orders = ["isin_0_carts",
               "emb_orders_v42",
               "count_dic_v19_p_chain_w_20_2_all_0",
               "count_dic_v14_p_chain_w_20_2_all_1",
               "count_dic_v16_p_chain_all_100_2_all_0",
              ]
    use_neg_rate_clicks = 0.2
    use_neg_rate_carts = 1.0
    use_neg_rate_orders = 1.0
    use_neg_rate = 1.0
if run_id == "v1":
    use_all_data_for_train = False
    param = {'lambda_l1': 2.3568477993751684, 'lambda_l2': 1.2824074933740028e-07, 'num_leaves': 17, 'feature_fraction': 0.8114194133521475, 'bagging_fraction': 0.8548900579828358, 'bagging_freq': 4, 'min_child_samples': 26}
    epochs = {"clicks":5000,"carts":5000,"orders":5000}
if run_id == "v2":
    use_all_data_for_train = False
    param = {'lambda_l1': 4.9893869833135405, 'lambda_l2': 4.075813767797564e-06, 'num_leaves': 28, 'feature_fraction': 0.6308923905874174, 'bagging_fraction': 0.8393970397682359, 'bagging_freq': 4, 'min_child_samples': 29}
    epochs = {"clicks":5000,"carts":5000,"orders":5000}
if run_id == "v3":
    use_all_data_for_train = False
    param = {'lambda_l1': 9.269025356408404, 'lambda_l2': 2.430580968028878e-06, 'num_leaves': 29, 'feature_fraction': 0.6269734474887927, 'bagging_fraction': 0.8546223203666866, 'bagging_freq': 4, 'min_child_samples': 30}
    epochs = {"clicks":5000,"carts":5000,"orders":5000}
if run_id == "v4":
    use_all_data_for_train = False
    param = {'lambda_l1': 2.7517491089268478, 'lambda_l2': 1.0119290348139934e-06, 'num_leaves': 69, 'feature_fraction': 0.6180786167340623, 'bagging_fraction': 0.8423715060704534, 'bagging_freq': 5, 'min_child_samples': 38}
    epochs = {"clicks":5000,"carts":5000,"orders":5000}
if run_id == "v5":
    use_all_data_for_train = False
    param = {'lambda_l1': 2.0584886213998326, 'lambda_l2': 1.6673942901727746e-05, 'num_leaves': 81, 'feature_fraction': 0.43917039108442923, 'bagging_fraction': 0.90377921125574, 'bagging_freq': 1, 'min_child_samples': 12}
    epochs = {"clicks":5000,"carts":5000,"orders":5000}
if run_id == "v6":
    use_all_data_for_train = False
    param = {'lambda_l1': 0.7380974372004203, 'lambda_l2': 2.2635347738867848e-05, 'num_leaves': 87, 'feature_fraction': 0.4096276999882601, 'bagging_fraction': 0.8455227207708554, 'bagging_freq': 1, 'min_child_samples': 15}
    epochs = {"clicks":5000,"carts":5000,"orders":5000}
if run_id == "v7":
    use_all_data_for_train = False
    param = {'lambda_l1': 8.821274864116514, 'lambda_l2': 2.3597036630027684e-06, 'num_leaves': 19, 'feature_fraction': 0.660133756381657, 'bagging_fraction': 0.7753512926841274, 'bagging_freq': 4, 'min_child_samples': 46}
    epochs = {"clicks":5000,"carts":5000,"orders":5000}
if run_id == "v8":
    use_all_data_for_train = True
    param = {'lambda_l1': 4.9893869833135405, 'lambda_l2': 4.075813767797564e-06, 'num_leaves': 28, 'feature_fraction': 0.6308923905874174, 'bagging_fraction': 0.8393970397682359, 'bagging_freq': 4, 'min_child_samples': 29}
    epochs = {"clicks":900,"carts":600,"orders":350}
if run_id == "v9":
    use_all_data_for_train = True
    param = {'lambda_l1': 0.7380974372004203, 'lambda_l2': 2.2635347738867848e-05, 'num_leaves': 87, 'feature_fraction': 0.4096276999882601, 'bagging_fraction': 0.8455227207708554, 'bagging_freq': 1, 'min_child_samples': 15}
    epochs = {"clicks":800,"carts":300,"orders":300}
    
# train and save models
label_type = "clicks"
train(*get_xy(dirs, feats_clicks, label_type, use_neg_rate = use_neg_rate * use_neg_rate_clicks, is_sample = is_sample),
     param, use_all_data_for_train, epochs
     )
label_type = "carts"
train(*get_xy(dirs, feats_carts, label_type, use_neg_rate = use_neg_rate * use_neg_rate_carts, is_sample = is_sample),
     param, use_all_data_for_train, epochs
     )
label_type = "orders"
train(*get_xy(dirs, feats_orders, label_type, use_neg_rate = use_neg_rate * use_neg_rate_orders, is_sample = is_sample),
     param, use_all_data_for_train, epochs
     )