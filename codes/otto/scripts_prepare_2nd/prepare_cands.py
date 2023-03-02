from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import sys
import glob
import os
from collections import defaultdict
import multiprocessing as mp


class MyDatasetCreater():
    """
    Class for generating candidates.
    The self.my_iter function is executed in the form of multiprocessing.
    In self.my_iter, receive the list of data paths, load and pass data to self.get_cands_from_df, and save the returned candidate data
    """

    def __init__(self, output_path, run_version):
        self.output_path = output_path
        self.run_version = run_version
        
    def my_iter(self, dirs):
        """
        Receive a list of data paths,
        compute candidate using self.get_cands_from_df,
        save the result
        """

        self.load()  # load model (ex. covisitation matrix)
        mapping = {"clicks": 1, "carts": 3, "orders": 6}
        for d in dirs:
            print(d)
            name = d.split("/")[-1].split(".")[0]
            df = pd.read_parquet(d)
            df["weight"] = df["type"].map(mapping)
            recs = self.get_cands_from_df(df)  # create candidates
            os.system(f"mkdir -p {self.output_path}")
            fn = self.output_path + "/{}.parquet".format(name)
            self.to_pandas(recs).to_parquet(fn) # save result
            print(f"save : {fn}")
    
    def to_pandas(self, data):
        recs = []
        for dd in tqdm(data):
            df = pd.DataFrame(dd[3], columns = dd[2])
            if len(df) == 0:
                df = pd.DataFrame([[dd[0], int(1e9)]], columns = ["session", "aid"]) ## dummy
            else:
                df["session"] = dd[0]
                df["aid"] = dd[1]
            recs.append(df)
        df = pd.concat(recs, axis=0)
        return df

    def load(self):
        """
        Load data required for generating candidates
        What to read depends on "run_version"
        """

        run_version = self.run_version
        
        # load a result from covisitation
        a = ["11_p_chain_full_20_5", "11_p_chain_full_100_5"]
        a += ["11_p_chain_all_20_2","12_p_chain_all_20_2","13_p_chain_all_20_2","14_p_chain_all_20_2","15_p_chain_all_20_2","16_p_chain_all_20_2", "17_p_chain_all_20_2"]
        a += [aa.replace("20", "100") for aa in a]
        a += ["11_p_chain_w_20_2", "11_p_chain_w_100_2"]
        a += ["18_p_chain_w_20_2", "18_p_chain_w_100_2"]
        a += ["14_p_chain_w_20_2", "14_p_chain_w_100_2"]
        a += ["19_p_chain_w_20_2", "19_p_chain_w_100_2"]
        a += ["20_p_chain_w_20_2", "20_p_chain_w_100_2"]
        if run_version in a:
            x = run_version.split("_")[0]
            self.count_dic = pd.read_pickle(f"../inputs/comatrix/count_dic_v{x}_all.pkl")
            
        # load two results from covisitation
        a = ["11_14_chain_1_2dic", "11_14_chain_2_2dic", "11_14_chain_3_2dic"]
        a += ["15_14_chain_1_2dic", "15_14_chain_2_2dic", "15_14_chain_3_2dic"]
        a += [c.replace("14", "13") for c in a] + [c.replace("14", "18") for c in a]
        if run_version in a:
            a = run_version.split("_")[0]
            b = run_version.split("_")[1]
            self.count_dic1 = pd.read_pickle(f"../inputs/comatrix/count_dic_v{a}_all.pkl")
            self.count_dic2 = pd.read_pickle(f"../inputs/comatrix/count_dic_v{b}_all.pkl")
        
        # load results from neural network
        a = ["emb_v15", "emb_v18", "emb_v21", "emb_v23",
            "emb_v27", "emb_v29", "emb_v31", "emb_v42"]
        if run_version in a:
            self.session2idx = {}
            self.aids = {}
            self.scores = {}
            n_ret_max = 200
            for label in ["clicks", "carts", "orders"]:
                mat, session2idx = pd.read_pickle("../inputs/emb_models/aid_h_{}.pkl".format(run_version.replace("emb", f"emb_{label}").replace("_rank", "").replace("_1000", "")))
                if "1000" in run_version:
                    n_ret_max = 1000
                n_middle = mat.shape[1]//2
                n_ret = min([n_ret_max, n_middle])
                self.session2idx[label] = session2idx
                self.aids[label] = mat[:,:n_middle][:,:n_ret]
                if "rank" in run_version:
                    self.scores[label] = mat[:,n_middle:][:,:n_ret]
                    self.scores[label] = self.scores[label] * 0 + (n_ret - np.arange(n_ret).reshape(1,-1))
                else:
                    self.scores[label] = mat[:,n_middle:][:,:n_ret]
    
    def get_from_count_dic_chain(self, aids, count_dic, n_iter, n_middle, name, use_weight):
        """
        apply covisitation at multiple time
        """

        n_ret = 100
        recs = []
        aids = aids[::-1]
        if use_weight:
            weights = np.log1p(np.arange(100))[::-1]
        else:
            weights = np.ones(max([len(aids)+1, 100]))
        
        for i in range(n_iter):
            # apply covisitation to aids series
            p_dict = defaultdict(int)
            for i_aid, src_aid in enumerate(aids):
                try:
                    dst_aids, counts = count_dic[src_aid]
                    s = sum(counts)
                    for aid, c in zip(dst_aids, counts):
                        p_dict[aid] += c/s*weights[i_aid]
                except:
                    pass
            if len(p_dict) == 0:
                for aid in aids:
                    p_dict[aid] = 1
            # get possible aids
            ps = sorted(p_dict.items(), key = lambda x : x[1], reverse = True)
            aids = [k for k, v in ps[:n_ret]]
            recs.append(
                pd.DataFrame(np.arange(len(aids))[::-1]+1, index = aids, columns = [name + "_" + str(i)])
            )

            # next source aids
            aids = [k for k, v in ps[:n_middle]]

        return recs
    
    
    def get_from_count_dic_chain_p_full_switch(self, aids, count_dic1, count_dic2, idx_switch, n_iter, n_middle, name):
        """
        apply covisitation at multiple time
        difference from "get_from_count_dic_chain" is switching covisitation when the specified iteration is reached
        """

        n_ret = 100
        recs = []
        count_dic = count_dic1
        for i in range(n_iter):
            # switch covisitation dict when the specified iteration is reached
            if i == idx_switch:
                count_dic = count_dic2
            p_dict = defaultdict(int)
            for src_aid in aids:
                try:
                    dst_aids, counts = count_dic[src_aid]
                    s = sum(counts)
                    for aid, c in zip(dst_aids, counts):
                        p_dict[aid] += c/s
                except:
                    pass
            if len(p_dict) == 0:
                for aid in aids:
                    p_dict[aid] = 1
            ps = sorted(p_dict.items(), key = lambda x : x[1], reverse = True)
            aids = [k for k, v in ps[:n_ret]]
            recs.append(
                pd.DataFrame(np.arange(len(aids))[::-1]+1, index = aids, columns = [name + "_" + str(i)])
            )
            aids = [k for k, v in ps[:n_middle]]

        return recs[-1:]
    

    def get_cands_from_df(self, df):
        """
        Takes a dataframe with aid, session, ts, type columns
        Returns candidates and scores
        run_version determines which strategy is used
        """

        self.run_version = run_version
        recs = []
        for i, (session, g_x) in enumerate(tqdm(df.groupby("session"))):            
            ds = []

            # aid visited in the session
            if run_version in ["0"]:
                types = pd.concat([g_x["type"] == "clicks", g_x["type"] == "carts", g_x["type"] == "orders"], axis = 1)
                types.columns = ["clicks", "carts", "orders"]
                g_x = pd.concat([g_x, types], axis = 1)
                g_x["ts_diff_pre"] = g_x["ts"].diff()//1000
                g_x["ts_diff_post"] = g_x["ts"].diff().shift(-1)//1000
                g_x["rank"] = np.arange(len(g_x))[::-1]
                g_x["count"] = 1
                agg = g_x.groupby("aid").agg({"ts_diff_pre" : "mean", "ts_diff_post" : "mean",
                                              "rank" : "min", "count" : "sum", "clicks" : "sum",
                                              "carts" : "sum", "orders" : "sum"}).fillna(0).astype(np.uint32)
                agg = agg.add_prefix(f"isin_{run_version}_")               
                ds.append(agg)
                    
            # candidates by covitation
            # use same covisitation at multiple time like beamsearch
            # use only last n aids in the session
            a = ["11_p_chain_full_20_5", "11_p_chain_full_100_5"]
            if run_version in a:
                sp = run_version.split("_")
                n_iter = int(sp[-1])
                n_middle = int(sp[-2])
                ran = [1,3,5,7,9]
                for k in ran:
                    _ds = self.get_from_count_dic_chain(g_x["aid"].to_list()[-k:],
                                                            self.count_dic, n_iter, n_middle,
                                                            f"count_dic_v{run_version}_{k}",
                                                            use_weight=False
                                                            )
                    ds += _ds

            # candidates by covisitation
            # use same covisitation at multiple time like beamsearch
            # use all aids in the session
            a = ["11_p_chain_all_20_2","12_p_chain_all_20_2","13_p_chain_all_20_2",
                 "14_p_chain_all_20_2","15_p_chain_all_20_2","16_p_chain_all_20_2",
                "17_p_chain_all_20_2"]
            a += [aa.replace("20", "100") for aa in a]
            if run_version in a:
                sp = run_version.split("_")
                n_iter = int(sp[-1])
                n_middle = int(sp[-2])
                _ds = self.get_from_count_dic_chain(g_x["aid"].to_list(),
                                                        self.count_dic, n_iter, n_middle,
                                                        f"count_dic_v{run_version}_all",
                                                        use_weight=False
                                                        )
                ds += _ds

            # candidates by covisitation
            # weighted by when it appeared in the session
            a = ["11_p_chain_w_20_2", "11_p_chain_w_100_2"]
            a += ["18_p_chain_w_20_2", "18_p_chain_w_100_2"]
            a += ["14_p_chain_w_20_2", "14_p_chain_w_100_2"]
            a += ["19_p_chain_w_20_2", "19_p_chain_w_100_2"]
            a += ["20_p_chain_w_20_2", "20_p_chain_w_100_2"]
            if run_version in a:
                sp = run_version.split("_")
                n_iter = int(sp[-1])
                n_middle = int(sp[-2])
                _ds = self.get_from_count_dic_chain(g_x["aid"].to_list(),
                                                        self.count_dic, n_iter, n_middle,
                                                        f"count_dic_v{run_version}_all",
                                                        use_weight=True
                                                        )
                ds += _ds

            # candidates by covisitation
            # apply two covisitations in order
            a = ["11_14_chain_1_2dic", "11_14_chain_2_2dic", "11_14_chain_3_2dic"]
            a += ["15_14_chain_1_2dic", "15_14_chain_2_2dic", "15_14_chain_3_2dic"]
            a += [c.replace("14", "13") for c in a] + [c.replace("14", "18") for c in a]
            if run_version in a:
                sp = run_version.split("_")
                switch_idx = int(run_version.split("_")[-2])
                _ds = self.get_from_count_dic_chain_p_full_switch(g_x["aid"].to_list(),
                                                               self.count_dic1,
                                                               self.count_dic2,
                                                                switch_idx,
                                                                switch_idx+1,20,
                                                               f"count_dic_v{run_version}_all")
                ds += _ds

            # caididates by nn
            # just load results by nn
            a = ["emb_v15", "emb_v18", "emb_v21", "emb_v23",
                  "emb_v27", "emb_v29", "emb_v31", "emb_v42"]
            if run_version in a:
                for label in ["clicks", "carts", "orders"]:
                    idx = self.session2idx[label][session]
                    d = pd.DataFrame(self.scores[label][idx], index = self.aids[label][idx], columns = [run_version.replace("emb", f"emb_{label}")])
                    ds.append(d)
                
            # store candidates and scores
            if len(ds) > 1:
                cands_with_score = pd.concat(ds, axis = 1)
            else:
                cands_with_score = ds[0]
            cands_all = list(cands_with_score.index)
            col_names = list(cands_with_score.columns)
            values = cands_with_score.fillna(0).values.astype(np.uint32)
            recs.append([session, cands_all, col_names, values])
        return recs


if __name__ == '__main__':
    run_version = sys.argv[1]  # A variable that determines which strategy to use to create a candidate

    class CFG:
        train_path = "../inputs/train_valid/train_parquet"
        val_path = "../inputs/train_valid/test_parquet"
        val_label_path = "../inputs/train_valid/test_labels.parquet"
        cands_path = f"../inputs/candidates/cands/{run_version}/"

    path = CFG.val_path
    out_path = CFG.cands_path
    print(path, out_path)
    n_worker = 7
    if "emb" in run_version:
        n_worker = 3
    dirs_splitted = np.array_split(glob.glob(path + "/*"), n_worker)
    processes = mp.Pool(n_worker)
    dataset = MyDatasetCreater(out_path, run_version)
    results = processes.map(dataset.my_iter, dirs_splitted)
    processes.close()
    processes.join()
