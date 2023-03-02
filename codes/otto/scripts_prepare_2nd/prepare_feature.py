from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import glob
from argparse import ArgumentParser
import os
import glob
import polars as pl

class Labels():
    """
    utility class for handle labels
    """

    def __init__(self):
        path_label = "../inputs/train_valid/test_labels.parquet"
        labels = pd.read_parquet(path_label)
        labels = labels.explode("ground_truth")
        labels["label_clicks"] = labels["type"] == "clicks"
        labels["label_carts"] = labels["type"] == "carts"
        labels["label_orders"] = labels["type"] == "orders"
        labels["aid"] = labels["ground_truth"]
        labels = labels[["session", "aid", "label_clicks", "label_carts", "label_orders"]]
        labels = labels.groupby(["session", "aid"]).sum().reset_index()
        labels = labels + 0
        labels = labels.reset_index(drop = True)
        labels["index"] = np.arange(len(labels))
        self.labels = labels[["aid", "label_clicks", "label_carts", "label_orders"]]
        self.session_indices_start_end = np.stack([
            labels.groupby("session")["index"].min().values,
            labels.groupby("session")["index"].max().values
        ]).T
        self.session2idx = {s:i for i,s in enumerate(labels["session"].unique())}
        
    def __call__(self, session):
        idx = self.session2idx[session]
        start_idx = self.session_indices_start_end[idx, 0]
        end_idx = self.session_indices_start_end[idx, 1]
        return self.labels.iloc[start_idx:end_idx+1]
        

def readout(path):
    """
    Read candidate data.
    Returns a list of lists of 4 values: session, candidate aids, score names, score values
    """

    df = pd.read_parquet(path)
    indices = np.where(df["session"] != df["session"].shift(1))[0]
    dss = []
    aids_all = df["aid"].to_list()
    df_ = df.drop(["aid", "session"], axis = 1)
    vs = df_.fillna(0).values
    cols = list(df_.columns)
    sessions = df["session"].unique()
    for i in range(len(indices)-1):
        v = vs[indices[i]:indices[i+1]]
        aids = aids_all[indices[i]:indices[i+1]]
        if int(1e9) in aids:
            dss.append([sessions[i], [], [], []])
        else:
            dss.append([sessions[i], aids, cols, v])
    aids = aids_all[indices[-1]:]
    v = vs[indices[-1]:]
    dss.append([sessions[-1], aids, cols, v])
    return dss


def create_data(i, n_data = 1e9):
    """
    Receive the data index and perform candidate join processing for the corresponding session
    """

    print(f"{i} th data")

    # load candidate data
    datas_list = []
    for dirs in dirs_list:
        print(dirs[i])
        datas = readout(dirs[i])
        datas_list.append(datas)
    
    # Combine candidates from multiple strategies
    # use numpy array for joining multiple candidates for speeding up
    train_data = []
    recall_rates = [0]
    n_cands_info_rec = []
    for i_session in tqdm(range(len(datas_list[0]))):
        if i_session > n_data: # for debugging
            break
        
        ## Get a complete picture of your data
        n_cols = 1 + 3
        col_names_all = ["aid"] + ['label_clicks', 'label_carts', 'label_orders']
        cands_all_set = set()
        sessions = []
        for i_feat, d in enumerate(datas_list):
            session, cands_all, col_names, values = d[i_session]
            cands_all_set.update(cands_all)
            n_cols += len(col_names)
            col_names_all += col_names
            sessions.append(session)
        n_cands = len(cands_all_set)
        n_cands_info_rec.append(n_cands)
        aid_map = {aid : i for i, aid in enumerate(cands_all_set)}
        assert(len(set(sessions)) == 1)

        ## set feature values
        v = np.zeros([n_cands, n_cols], dtype = int)
        i_col = 1 + 3
        for i_feat, d in enumerate(datas_list):
            session, cands_all, col_names, values = d[i_session]
            indices_row = np.array([aid_map[aid] for aid in cands_all])
            if len(indices_row) != 0:
                v[indices_row, i_col : i_col + len(col_names)] = values + 1
                v[indices_row, 0] = cands_all
            i_col += len(col_names) 

        ## join label information
        if args.with_label:
            label = label_loader(session)
            use_labels = np.array([[aid_map[v[0]], *v[1:]] for v in label.values if v[0] in aid_map])
            recall = len(use_labels) / len(label)
            recall_rates.append(recall)
            
            if len(use_labels) > 0:
                v[use_labels[:,0], 1:4] = use_labels[:,1:]

        ## store candidates and feature values of the session
        j = pl.DataFrame(v, columns = col_names_all) 
        j = j.with_columns(pl.lit(session).alias("session"))
        train_data.append(j)

        ## if data size reaches the threshold, return the data to save for memory reduction
        if i_session > 0 and i_session % 5000 == 0:
            df = concat_safe(train_data)
            print("recall rate : ", np.mean(recall_rates))
            print("num cands : ", np.mean(n_cands_info_rec))
            yield df
            train_data = []
    df = concat_safe(train_data)
    yield df

def concat_safe(train_data):
    """
    Concat dataframes while keeping the column order same
    """
    train_data_fill = []
    cols_all = set([c for j in train_data for c in j.columns])
    for j in train_data:
        if len(j.columns) != len(cols_all):
            cols = [c for c in cols_all if c not in j.columns]
            j = j.with_columns([pl.lit(0).alias(c).cast(pl.Int64) for c in cols])
        train_data_fill.append(j.select(cols_all))
    df = pl.concat(train_data_fill)
    return df


def my_iter(indices):
    """
    Receive the data index list and turn the processing loop
    """
    if args.with_additinal_feats:
        addfeats = AddFeats()
    for i in indices:
        fn = dirs_list[0][i].split("/")[-1]
        print("fn = ", fn)
        os.system("mkdir -p " + CFG.output_path_val)
        fn = fn.replace("pkl", "parquet")
        for k, df in enumerate(create_data(i)):
            if args.with_additinal_feats:
                df = addfeats.to_uint32(df)
                df = addfeats.add(df)  # Add features unrelated to candidate generation strategy
            df = df.select(sorted(df.columns))
            path = CFG.output_path_val + "/" + str(k) + "_" + fn
            print("save! : ", path)
            df.write_parquet(path)
        
class AddFeats():
    """
    class for adding features unrelated to candidate generation strategy
    """

    def __init__(self):
        # features about aid
        aid_feats = pl.read_parquet("../inputs/stats/aid_feats.parquet")
        aid_feats2 = pl.read_parquet("../inputs/stats/aid_feats2.parquet")
        aid_feats3 = pl.read_parquet("../inputs/stats/aid_feats3.parquet")

        # features about aid and last time of session
        h_aid_feats = pl.read_parquet("../inputs/stats/h_aid_rate.parquet")
        h_aid_feats2 = pl.read_parquet("../inputs/stats/h_aid_rate_v2.parquet")

        # Information that links the last time of the session to session id
        session2h = pd.read_pickle("../inputs/stats/session2h.pkl")
        self.session2h = pl.DataFrame(session2h, columns = ["session", "h"])

        # features about session
        self.session_feats = pl.read_parquet("../inputs/stats/session_feats.parquet")

        # cast
        self.to_uint32(aid_feats)
        self.to_uint32(aid_feats2)
        self.to_uint32(aid_feats3)
        self.to_uint32(h_aid_feats)
        self.to_uint32(h_aid_feats2)
        self.to_uint32(self.session2h)
        self.to_uint32(self.session_feats)
        
        # aid feats
        self.aid_feats = aid_feats.join(aid_feats2, on = "aid", how = "outer").join(aid_feats3, on = "aid", how = "outer")
        # aid x hh feats
        self.h_aid_feats = h_aid_feats.join(h_aid_feats2, on = ["aid", "h"], how = "outer")
        
        # Specify the column that performs ranking processing
        self.rank_cols = ['a_count', 'a_num_carts', 'a_num_clicks', 'a_num_orders', 'a_rate_carts', 'a_rate_clicks',
                     'a_rate_orders', 'a_session_nunique',
                     'ah_popular_-1h_all', 'ah_popular_-1h_carts', 'ah_popular_-1h_clicks', 'ah_popular_-1h_orders',
                     'ah_popular_1h_all', 'ah_popular_1h_carts', 'ah_popular_1h_clicks', 'ah_popular_1h_orders']
        self.rank_cols += ['ah_popular_24_False_all', 'ah_popular_24_False_carts', 'ah_popular_24_False_clicks', 'ah_popular_24_False_orders', 'ah_popular_24_True_all', 'ah_popular_24_True_carts', 'ah_popular_24_True_clicks', 'ah_popular_24_True_orders', 'ah_popular_48_False_all', 'ah_popular_48_False_carts', 'ah_popular_48_False_clicks', 'ah_popular_48_False_orders', 'ah_popular_48_True_all', 'ah_popular_48_True_carts', 'ah_popular_48_True_clicks', 'ah_popular_48_True_orders', 'ah_popular_6_False_all', 'ah_popular_6_False_carts', 'ah_popular_6_False_clicks', 'ah_popular_6_False_orders', 'ah_popular_6_True_all', 'ah_popular_6_True_carts', 'ah_popular_6_True_clicks', 'ah_popular_6_True_orders']
        
        _rank_cols = ['a_count', 'a_num_carts', 'a_num_clicks', 'a_num_orders', 'a_rate_carts', 'a_rate_clicks', 'a_rate_orders', 'a_session_nunique']
        rank_cols = [c + "2" for c in _rank_cols] + [c + "3" for c in _rank_cols]
        self.rank_cols += rank_cols
        
        # Specify the column not used
        self.drops = ['ah_popular_-1h_orders',
 'count_dic_v11_p_chain_full_100_5_3_2',
 'count_dic_v11_p_chain_full_100_5_5_1',
 'count_dic_v11_p_chain_full_100_5_5_2',
 'count_dic_v11_p_chain_full_100_5_7_0',
 'count_dic_v11_p_chain_full_100_5_7_1',
 'count_dic_v11_p_chain_full_100_5_7_2',
 'count_dic_v11_p_chain_full_100_5_9_1',
 'count_dic_v11_p_chain_full_20_5_7_0',
 'count_dic_v11_p_chain_full_20_5_7_1',
 'count_dic_v11_p_chain_full_20_5_7_2',
 'count_dic_v11_p_chain_full_20_5_9_0',
 'count_dic_v11_p_chain_full_20_5_9_1',
 'count_dic_v11_p_chain_full_20_5_9_2'] + ['ah_popular_-1h_all',
 'count_dic_v11_p_chain_full_100_5_3_1',
 'count_dic_v11_p_chain_full_100_5_9_0',
 'count_dic_v11_p_chain_full_20_5_5_2',
 'count_dic_v12_p_chain_all_20_2_all_0',
 'count_dic_v16_p_chain_all_100_2_all_1'] + ['ah_popular_1h_orders',
 'count_dic_v11_p_chain_all_100_2_all_1',
 'count_dic_v11_p_chain_all_20_2_all_0',
 'count_dic_v11_p_chain_full_100_5_5_0',
 'count_dic_v11_p_chain_full_100_5_9_2',
 'count_dic_v11_p_chain_full_20_5_5_0',
 'count_dic_v11_p_chain_full_20_5_5_1'] + ['ah_popular_-1h_all_12',
 'ah_popular_-1h_all_4',
 'ah_popular_-1h_carts_4',
 'ah_popular_-1h_clicks_12',
 'ah_popular_-1h_orders_12',
 'ah_popular_-1h_orders_4',
 'ah_popular_-1h_orders_4_rank',
 'ah_popular_1h_all_12',
 'ah_popular_1h_all_12_rank',
 'ah_popular_1h_all_4',
 'ah_popular_1h_carts_12',
 'ah_popular_1h_carts_4',
 'ah_popular_1h_clicks_12',
 'ah_popular_1h_clicks_12_rank',
 'ah_popular_1h_orders_12',
 'ah_popular_1h_orders_4',
 'count_dic_v11_p_chain_full_20_5_3_0'] + ['a_num_orders',
 'ah_popular_-1h_carts_12_rank',
 'ah_popular_-1h_clicks_4',
 'ah_popular_1h_carts_12_rank',
 'ah_popular_1h_clicks_4',
 'ah_popular_1h_orders_4_rank',
 'count_dic_v11_p_chain_all_20_2_all_1',
 'count_dic_v11_p_chain_w_20_2_all_0',
 'count_dic_v11_p_chain_w_20_2_all_1',
 'count_dic_v13_p_chain_all_20_2_all_0',
 'count_dic_v16_p_chain_all_20_2_all_0',
 'count_dic_v18_p_chain_w_20_2_all_0'] + ['ah_popular_-1h_all_12_rank',
 'ah_popular_-1h_clicks_4_rank',
 'ah_popular_-1h_orders_12_rank',
 'ah_popular_1h_carts_4_rank',
 'ah_popular_1h_orders_rank',
 'isin_0_clicks']
        
    def to_uint32(self, df):
        return df.with_columns(
            [pl.col(c).cast(pl.UInt32).alias(c) for c in df.columns]
        )
        
    def add(self, df):
        """
        Receive dataframe with sessions and candidate aids, add features and return
        """

        pre_shape = df.shape
        print("add feats")
        
        # Discard features with overlapping names (for safe)
        original_cols = df.columns
        drs = [c for c in self.aid_feats.columns if c in original_cols]
        drs += [c for c in self.h_aid_feats.columns if c in original_cols]
        drs = [c for c in drs if c not in ["aid", "session"]]
        df = df.drop(drs)
        
        # join features
        df = (df.join(self.aid_feats, on = "aid", how = "left"))
        df = (df.join(self.session2h, on = "session", how = "left")
                .join(self.h_aid_feats, on = ["aid", "h"], how = "left")
                .drop("h"))
        df = (df.join(self.session_feats, on = "session", how = "left"))
        
        # rank processing
        df = df.with_columns(
            [
                ((-pl.col(c)).rank().over("session")).cast(pl.UInt32).alias(c + "_rank") for c in self.rank_cols
            ]
        )
        df = df.fill_null(0)

        # drop useless feats
        df = df.drop([c for c in self.drops if c in df.columns])
        print(f"{pre_shape} -> {df.shape}")
        return df
    
    
parser = ArgumentParser()
parser.add_argument("--data_idx", type=int, default=0, help = "the first idx of the data to process")
parser.add_argument("--data_idx_end", type=int, default=20, help = "the last idx of the data to process")
parser.add_argument("--with_label", action='store_true', help = "1 for train data and 0 for test data")
parser.add_argument("--with_additinal_feats", action='store_true', help = "join feats which is not used in caididate geneartion")
parser.add_argument("--simple", action='store_true', help = "use small features for simple solution")
args = parser.parse_args()

class CFG:
    cands_path = f"../inputs/candidates/cands/"
    output_path_val = f"../inputs/2nd_stage/"

if args.with_label:
    label_loader = Labels()

# Specify which candidates to use
if args.simple:
    use_cands = [
         '0',
         '14_p_chain_w_20_2',
         '14_p_chain_all_20_2',
         '15_p_chain_all_20_2',
         '16_p_chain_all_100_2',
         '19_p_chain_w_20_2',
         'emb_v42',
    ]
else:
    use_cands = [
         '0',
         '11_p_chain_full_20_5',
         '11_p_chain_full_100_5',
         '11_p_chain_all_20_2',
         '12_p_chain_all_20_2',
         '13_p_chain_all_20_2',
         '14_p_chain_all_20_2',
         '15_p_chain_all_20_2',
         '16_p_chain_all_20_2',
         '17_p_chain_all_20_2',
         '11_p_chain_all_100_2',
         '12_p_chain_all_100_2',
         '13_p_chain_all_100_2',
         '14_p_chain_all_100_2',
         '15_p_chain_all_100_2',
         '16_p_chain_all_100_2',
         '17_p_chain_all_100_2',
         "11_14_chain_1_2dic",
         "11_14_chain_2_2dic",
         "11_14_chain_3_2dic",
        "11_p_chain_w_20_2",
        "11_p_chain_w_100_2",
        "15_14_chain_1_2dic",
        "15_14_chain_2_2dic",
        "15_14_chain_3_2dic",
        "15_13_chain_1_2dic",
        "15_13_chain_2_2dic",
        "15_13_chain_3_2dic",
        "11_13_chain_1_2dic",
        "11_13_chain_2_2dic",
        "11_13_chain_3_2dic",
        "14_p_chain_w_20_2",
        "14_p_chain_w_100_2",
        "18_p_chain_w_20_2",
        "18_p_chain_w_100_2",
        "15_18_chain_1_2dic",
        "15_18_chain_2_2dic",
        "15_18_chain_3_2dic",
        "11_18_chain_1_2dic",
        "11_18_chain_2_2dic",
        "11_18_chain_3_2dic",
        "19_p_chain_w_20_2",
        "20_p_chain_w_20_2",
         'emb_v15',
         'emb_v18',
         'emb_v21',
         'emb_v23',
         'emb_v27',
         'emb_v29',
         'emb_v31',
         'emb_v42',
    ]
path_list = [CFG.cands_path+"/"+c for c in use_cands]
dirs_list = [glob.glob(path + "/*") for path in path_list]
print([[d[0], len(d)] for d in dirs_list])

print("start creating data for reranker")
data_idx_end = min([len(dirs_list[0]), args.data_idx_end])
indices = np.arange(args.data_idx, data_idx_end)
my_iter(indices)