from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import sys
import time
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import glob
from argparse import ArgumentParser
import os
import polars as pl

version = __file__.split("_")[-1].split(".")[0]
class CFG:
    train_path = "../inputs/train_valid/train_parquet"
    val_path = "../inputs/train_valid/test_parquet"
    val_label_path = "../inputs/train_valid/test_labels.parquet"
    test_path = "../inputs/parquet/test_parquet/"
    output_path = "../inputs/emb_models/"
    output_fn = f"aid_h_emb_{version}.pkl"
    output_model_fn = f"model_emb_{version}.pkl"
    n_epoch = 5
    n_length = 6
    n_feat = 4 ## aidを除く， hのsin, cosも除く
    n_aid = 1855612
    temperature = 0.05
    batch_size = 1000 * torch.cuda.device_count()
    n_topk = 200
    emb_dim = 300
    time_emb_dim = 50
    n_label = 5
    lr = 0.001 * torch.cuda.device_count()
    h_min = 460918
    h_max = 461757 + 3
    d_min = 19204
    d_max = 19239 + 3
    run_test = False
    n_neg_coef = 10
    
    
if CFG.run_test:
    CFG.n_epoch = 1
    # run_test = False#True
CFG.x_cols  = [f"aid_pre{i}" for i in range(CFG.n_length)] 
CFG.x_cols += [f"h_pre{i}" for i in range(CFG.n_length)]
CFG.x_cols += [f"d_pre{i}" for i in range(CFG.n_length)]
CFG.x_cols += [f"weight_pre{i}" for i in range(CFG.n_length)]
CFG.x_cols += [f"ts_pre{i}" for i in range(CFG.n_length)]
    
def metric_smoothing(mets, mets_smooth):
    assert(len(mets) == len(mets_smooth))
    metric_smooth_coef = 0.99
    met_str = []
    for k, (key, value) in enumerate(mets_smooth.items()):
        try:
            v = mets[k].item()
        except:
            v = mets[k]
        if mets_smooth[key] is None:
            mets_smooth[key] = v
        else:
            mets_smooth[key] = mets_smooth[key] * metric_smooth_coef + v * (1 - metric_smooth_coef)
        met_str.append("{} : {:.4f} ".format(key, mets_smooth[key]))
    met_str = ", ".join(met_str)
    return met_str, mets_smooth

def preprocess_df(df, with_label):
    df = pl.from_pandas(df)
    df = df.sort(["session", "ts"])
    n_length = CFG.n_length
    n_label = CFG.n_label
    n_aid = CFG.n_aid
    assert(n_aid > df["aid"].max()+3)

    weights = {"clicks" : 1, "carts" : 3, "orders" : 6}
    mapper = pl.DataFrame({
        "type" : list(weights.keys()),
        "weight": list(weights.values())
    })
    df = df.join(mapper, on = "type")
    df = df.with_columns((pl.col("ts")//3600//1000).alias("h"))
    df = df.with_columns((pl.col("h") - CFG.h_min).alias("h"))
    df = df.with_columns((pl.col("ts")//3600//1000//24).alias("d"))
    df = df.with_columns((pl.col("d") - CFG.d_min).alias("d"))

    exprs = []
    for i in range(0,n_length):
        exprs += [
            pl.col("aid").shift(i).alias(f"aid_pre{i}"),
            pl.col("h").shift(i).alias(f"h_pre{i}"),
            pl.col("d").shift(i).alias(f"d_pre{i}"),
            pl.col("weight").shift(i).alias(f"weight_pre{i}"),
            ((pl.col("ts") - pl.col("ts").shift(i)) // 1000).alias(f"ts_pre{i}"),
            pl.col("session").shift(i).alias(f"session_pre{i}"),
        ]

    df = df.with_columns(exprs)

    def get_replace(name, rep, condition):
        return pl.when(
            condition).then(
            pl.lit(rep)
        ).otherwise(pl.col(name)).alias(name)

    conditions = [pl.col("session") != pl.col(f"session_pre{i}") for i in range(n_length)]

    df = df.with_columns(
        [get_replace(f"aid_pre{i}", n_aid - 1, conditions[i]) for i in range(n_length)] + \
        [get_replace(f"h_pre{i}", CFG.h_max-CFG.h_min, conditions[i]) for i in range(n_length)] + \
        [get_replace(f"d_pre{i}", CFG.d_max-CFG.d_min, conditions[i]) for i in range(n_length)] + \
        [get_replace(f"weight_pre{i}", 0, conditions[i]) for i in range(n_length)] + \
        [get_replace(f"ts_pre{i}", -1, conditions[i]) for i in range(n_length)]
    )

    if with_label:
        for i in range(0, n_label):
            exprs = [pl.col(["aid"]).shift(-1-i).alias(f"aid_label"),
                     pl.col(["weight"]).shift(-1-i).alias(f"weight_label")]
            df = df.with_columns(exprs)
            condition = pl.col("session") == pl.col("session").shift(-1-i)
            for k in range(0,n_length):
                condition = condition & (pl.col(f"aid_pre{k}") != df["aid_label"])

            if i == 0:
                df_all = df.filter(condition)
            else:
                df_all = pl.concat([df_all, df.filter(condition)])
            print(df_all.shape)
        return torch.LongTensor(df_all[CFG.x_cols + ["weight_label", "aid_label"]].to_numpy())
    return df.to_pandas()

def train(model, opt, xy):
    mets_smooth = {"loss" : None, "acc" : None}
    model.train()
    batch_size = CFG.batch_size
    batch_indices = np.array_split(np.random.permutation(len(xy)), len(xy)//batch_size)
    i = 0
    mets_smooth_acc_rec = [0]
    train_end_flag = False
    for indices in tqdm(batch_indices):
        xy_batch = xy[indices]
        outs = model(xy_batch)
        outs = [torch.mean(l) for l in outs]
        loss, acc = outs
        model.zero_grad()
        loss.backward()
        opt.step()

        i+=1
        mets_str, mets_smooth = metric_smoothing([loss,
                                                  acc,
                                                 ], mets_smooth)
        if i%100 == 0:
            print(i, mets_str)
            if CFG.run_test & (i > 1000):
                break
        if i%2000 == 0:
            if mets_smooth["acc"] < mets_smooth_acc_rec[-1]:
                train_end_flag = opt.step_scheduler()
                if train_end_flag:
                    break
            mets_smooth_acc_rec.append(mets_smooth["acc"])            
            print("acc improve {} -> {}".format(mets_smooth_acc_rec[-2], mets_smooth_acc_rec[-1]))
    return model, train_end_flag

def inference(model, is_inference_all):
    print("start inference")
    model = model.eval()
    n_topk = CFG.n_topk
    ## parquetからの読み込みと処理
    if is_inference_all:
        path = CFG.val_path
        print("inference on all data")
    else:
        path = glob.glob(CFG.val_path + "/*")[0]
        print("inference on partial data")
    df_inf = pd.read_parquet(path)
    df_inf = preprocess_df(df_inf, with_label=False)
    df_inf = df_inf.groupby("session").last()

    x_cols = CFG.x_cols
    sessions = np.array(list(df_inf.index))
    x_tensor = torch.LongTensor(df_inf[x_cols].values)

    ## inference
    batch_size = 50
    weights = {"clicks" : 1, "carts" : 3, "orders" : 6}
    aid_cands = torch.arange(CFG.n_aid, device = device)
    # prepare aid embedding in advance
    y_em = model.do_emb(aid_cands).detach()
    y_em = y_em/torch.norm(y_em, dim=1, keepdim = True)
    batch_indices = np.array_split(np.random.permutation(len(x_tensor)), len(x_tensor)//batch_size)
    for label, weight in weights.items(): # predict for each type
        label_type = (torch.zeros(batch_size*2, device = device) + weight).long()
        recs = []
        for indices in tqdm(batch_indices):
            x_batch = x_tensor[indices].to(device)
            # session embedding for the label_type
            x_em = model.calc_query_emb(x_batch, label_type[:len(x_batch)])
            x_em = x_em.detach()
            x_em = x_em/torch.norm(x_em, dim=1, keepdim = True)

            # cosine similarity between session embedding and aid embedding
            cossim = torch.matmul(x_em, y_em.T)

            # get topk aids and scores
            scores, topk = torch.topk(cossim, n_topk, axis = 1)
            scores = scores.cpu().numpy()
            scores = (scores * 1000).astype(int)
            topk = topk.cpu().numpy()    
            ses = sessions[indices]
            recs.append([ses, topk, scores])

            if CFG.run_test:
                if len(recs) > 5:
                    break
                    
        ## 推論結果まとめ
        sessions_save = np.hstack([r[0] for r in recs])
        topks = np.vstack([r[1] for r in recs])
        scores = np.vstack([r[2] for r in recs])

        lis = []
        dic = {}
        lis = np.zeros([len(sessions_save), n_topk*2], dtype = np.int32)
        for i in tqdm(range(len(sessions_save))):
            lis[i] = list(topks[i]) + list(scores[i])
            # lis.append([list(topks[i]) + list(scores[i])])
            dic[sessions_save[i]] = i
        lis = np.array(lis).astype(np.uint32)
        os.system(f"mkdir -p {CFG.output_path}")
        dst = CFG.output_path + CFG.output_fn.replace("emb", f"emb_{label}")
        pd.to_pickle([lis, dic], dst)
        print("save! : ", dst)
        import gc; gc.collect()



class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.emb = torch.nn.Embedding(CFG.n_aid, CFG.emb_dim, padding_idx=CFG.n_aid-1,sparse=True)
        # self.emb_x = torch.nn.Embedding(CFG.n_aid, CFG.emb_dim, padding_idx=CFG.n_aid-1,sparse=True)
        self.emb_h = torch.nn.Embedding(CFG.h_max-CFG.h_min+2, CFG.time_emb_dim, padding_idx=CFG.h_max-CFG.h_min)
        self.emb_d = torch.nn.Embedding(CFG.d_max-CFG.d_min+2, CFG.time_emb_dim, padding_idx=CFG.d_max-CFG.d_min)
        inp_dim = CFG.emb_dim * CFG.n_length + (CFG.n_feat + CFG.time_emb_dim * 2) * CFG.n_length + 1
        self.mlp = torch.nn.Sequential(
            torch.nn.BatchNorm1d(inp_dim),
            torch.nn.Linear(inp_dim,inp_dim),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(inp_dim),
            torch.nn.Linear(inp_dim,inp_dim//2),
            torch.nn.GELU(),
            torch.nn.BatchNorm1d(inp_dim//2),
            torch.nn.Linear(inp_dim//2,CFG.emb_dim)
        )
        nn.init.uniform_(self.emb.weight, -1.0, 1.0)
        nn.init.uniform_(self.emb_h.weight, -1.0, 1.0)
        nn.init.uniform_(self.emb_d.weight, -1.0, 1.0)
        
        
    def forward(self, xy_batch):
        x_batch = xy_batch[:,:-2]
        label_type = xy_batch[:,-2]
        y_batch = xy_batch[:,-1]
        neg_aids = torch.randperm(CFG.n_aid, device = xy_batch.get_device())[:CFG.n_neg_coef * len(xy_batch)]

        x_em = self.calc_query_emb(x_batch, label_type)
        y_em = self.do_emb(y_batch)
        neg_em = self.do_emb(neg_aids)
        # y_em = torch.cat([y_em, neg_em])

        x_em_norm = torch.norm(x_em, dim=1, keepdim = True)
        x_em = x_em/x_em_norm
        y_em = y_em/torch.norm(y_em, dim=1, keepdim = True)
        neg_em = neg_em/torch.norm(neg_em, dim=1, keepdim = True)
        
        cossim_pos = torch.sum(x_em * y_em, dim = 1, keepdim = True)
        
        cossim_neg = torch.matmul(x_em, neg_em.T)
        cossim = torch.cat([cossim_pos, cossim_neg], dim = 1) / CFG.temperature
        p = torch.softmax(cossim,dim=1)
        labels = torch.arange(len(p), device = xy_batch.get_device())
        loss = - torch.mean(label_type * torch.log(p[:,0]))
        acc = torch.mean((torch.max(p,dim=1)[1] == 0).to(torch.float))
        return loss, acc

        
    def calc_query_emb(self, x_batch, label_type):
        
        x_batch = x_batch.reshape(-1, CFG.n_feat + 1, CFG.n_length).transpose(1,2) ## (batch_size, n_length, n_feat)
        em = self.emb(x_batch[:,:,0])
        x_other = torch.cat([
            torch.sin(x_batch[:,:,1:2] % 24 / 24 * np.pi * 2),
            torch.cos(x_batch[:,:,1:2] % 24 / 24 * np.pi * 2),
            x_batch[:,:,3:],
        ], dim=2)
        emh = self.emb_h(x_batch[:,:,1])
        emd = self.emb_d(x_batch[:,:,2])
        x = torch.cat([em, x_other, emh, emd], dim = 2)
        x = x.reshape(len(x), -1)
        x = torch.cat([x, label_type.unsqueeze(1)], dim = 1)
        return self.mlp(x)
    
    def do_emb(self, x):
        return self.emb(x)
    
class MyOptimizer():
    def __init__(self, model):
        params = []
        params_emb = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name in ["module.emb.weight", "module.emb_x.weight"]:
                    params_emb.append(param)
                else:
                    params.append(param)
                    
        self.opt = torch.optim.AdamW(params, lr = CFG.lr)
        self.opt_em = torch.optim.SparseAdam(params_emb, lr = CFG.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.opt, step_size=1, gamma=0.5)
        self.scheduler_em = torch.optim.lr_scheduler.StepLR(self.opt_em, step_size=1, gamma=0.5)
        self.count = 0
        
    def step(self):
        self.opt.step()
        self.opt_em.step()
        
    def step_scheduler(self):
        
        self.scheduler.step()
        self.scheduler_em.step()
        self.count += 1
        print("lr half", self.count)
        if self.count > 4:
            return True
        else:
            return False

device = "cuda"
parser = ArgumentParser()
parser.add_argument("--pretrain", default="")
parser.add_argument("--is_skip_train", action='store_true')
parser.add_argument("--is_inference_all", action='store_true')
args = parser.parse_args()

# python emb_model_${v}.py
# python emb_model_${v}.py --pretrain at_train --is_skip_train --is_inference_all
# python emb_model_${v}.py --pretrain at_test --is_inference_all

if args.pretrain == "at_train":
    model_path = CFG.output_path + "/last_"+CFG.output_model_fn
elif args.pretrain == "at_test":
    model_path = "../../otto/" + CFG.output_path.replace("..", "") + "/last_"+CFG.output_model_fn
elif args.pretrain == "":
    model_path = None


if not args.is_skip_train:
    df = pd.concat([
        pd.read_parquet(CFG.train_path),
        pd.read_parquet(CFG.val_path),
        # pd.read_parquet(CFG.test_path),
    ], axis = 0).reset_index(drop=True)
    df = df[df["ts"] > (df["ts"].max() - (3600 * 24 * 14 * 1000))]
    if CFG.run_test:
        df = df[:10000000]
    print(df.shape)

    xy = preprocess_df(df, with_label=True)
    del df
    import gc; gc.collect()
    print(xy.shape)

if model_path is None:
    model = Encoder()
else:
    print("load model from ", model_path)
    model = pd.read_pickle(model_path).module

model = torch.nn.DataParallel(model) # make parallel
torch.backends.cudnn.benchmark = True
model.do_emb = model.module.do_emb
model.calc_query_emb = model.module.calc_query_emb
model = model.to(device)

if not args.is_skip_train:
    opt = MyOptimizer(model)
    os.system(f"mkdir -p {CFG.output_path}")
    for i in range(CFG.n_epoch):
        model, train_end_flag = train(model, opt, xy)
        pd.to_pickle(model.cpu(), CFG.output_path + "/" + str(i)+"_"+CFG.output_model_fn)
        model = model.to(device)
        if train_end_flag:
            break
    pd.to_pickle(model.cpu(), CFG.output_path + "/" + "last"+"_"+CFG.output_model_fn)
    model = model.to(device)
inference(model, args.is_inference_all)