from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import torch
from torch import nn
import glob
from argparse import ArgumentParser
import os
import polars as pl

version = __file__.split("_")[-1].split(".")[0]
class CFG:
    train_path = "../inputs/train_valid/train_parquet"
    val_path = "../inputs/train_valid/test_parquet"
    output_path = "../inputs/emb_models/"
    output_fn = f"aid_h_emb_{version}.pkl"
    output_model_fn = f"model_emb_{version}.pkl"
    n_epoch = 10 
    n_length = 10  # input length
    n_feat = 2  # num of features except for aid
    temperature = 0.05
    batch_size = 2000 * max([torch.cuda.device_count(), 1])
    n_topk = 200 # on inference, how many topk samples are saved
    emb_dim = 500
    time_emb_dim = 50
    n_label = 10  # label length
    lr = 0.001 * max([torch.cuda.device_count(), 1])
    n_aid = 1855612
    h_min = 460918
    h_max = 461757 + 3
    d_min = 19204
    d_max = 19239 + 3
    n_neg_coef = 30 # n_neg_coef * batch_size is the number of negative aids 
    neg_topk = 1000 # the number of negative aids which is used for train
    run_test = False
    
if CFG.run_test:
    CFG.n_epoch = 1

# columns used as inputs
CFG.x_cols  = [f"aid_pre{i}" for i in range(CFG.n_length)] 
CFG.x_cols += [f"h_pre{i}" for i in range(CFG.n_length)]
CFG.x_cols += [f"weight_pre{i}" for i in range(CFG.n_length)] # type of aid

# columns used as targets
CFG.y_cols  = [f"aid_label{i}" for i in range(CFG.n_label)]
CFG.y_cols += [f"weight_label{i}" for i in range(CFG.n_label)] # type of aid

def metric_smoothing(mets, mets_smooth):
    """
    utility for displaying metrics during training
    Receive the value of the metric at that time,
    returns the value of metric after smoothing
    """
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
    """
    Format data to suit nn input
    output is Dataframe with CFG.x_cols + CFG.y_cols
    """
    df = pl.from_pandas(df)
    df = df.sort(["session", "ts"])
    n_length = CFG.n_length
    n_label = CFG.n_label
    n_aid = CFG.n_aid
    assert(n_aid > df["aid"].max()+3)

    # mapping : type string -> weight int
    weights = {"clicks" : 1, "carts" : 3, "orders" : 6}
    mapper = pl.DataFrame({
        "type" : list(weights.keys()),
        "weight": list(weights.values())
    })
    df = df.join(mapper, on = "type")

    # hour information from ts
    df = df.with_columns((pl.col("ts")//3600//1000).alias("h"))
    df = df.with_columns((pl.col("h") - CFG.h_min).alias("h"))

    # shift features to get past values
    exprs = []
    for i in range(0,n_length):
        exprs += [
            pl.col("aid").shift(i).alias(f"aid_pre{i}"),
            pl.col("h").shift(i).alias(f"h_pre{i}"),
            pl.col("weight").shift(i).alias(f"weight_pre{i}"),
            pl.col("session").shift(i).alias(f"session_pre{i}"),
        ]
    df = df.with_columns(exprs)
    def get_replace(name, rep, condition):
        return pl.when(
            condition).then(
            pl.lit(rep)
        ).otherwise(pl.col(name)).alias(name)

    # If the session id is different before and after shifting, replace features to pre defined non informative values
    # aid -> n_aid - 1
    # h -> CFG.h_max-CFG.h_min+3
    # weight -> 0
    conditions = [pl.col("session") != pl.col(f"session_pre{i}") for i in range(n_length)]
    df = df.with_columns(
        [get_replace(f"aid_pre{i}", n_aid - 1, conditions[i]) for i in range(n_length)] + \
        [get_replace(f"h_pre{i}", CFG.h_max-CFG.h_min+3, conditions[i]) for i in range(n_length)] + \
        [get_replace(f"weight_pre{i}", 0, conditions[i]) for i in range(n_length)]
    )

    # for training
    if with_label:
        # get aids and weights in future as labels
        for i in range(0, n_label):
            exprs = [
                pl.col(["aid"]).shift(-1-i).alias(f"aid_label{i}"),     
                pl.col(["weight"]).shift(-1-i).alias(f"weight_label{i}")
            ]
            # if session is disagreement, replace labels with predefined values
            condition = pl.col("session") != pl.col("session").shift(-1-i)
            exprs_rep = [
                get_replace(f"aid_label{i}", n_aid - 1, condition).fill_null(n_aid - 1),
                get_replace(f"weight_label{i}", 0, condition).fill_null(0),
            ]
            df = df.with_columns(exprs).with_columns(exprs_rep)

        # "aid_label0 != n_aid -1" indicates that there is at least one label
        condition = pl.col("aid_label0") != n_aid - 1
        return torch.LongTensor(df.filter(condition)[CFG.x_cols + CFG.y_cols].to_numpy())
    return df.to_pandas()

def train(model, opt, xy):
    """
    train nn model for one epoch

    model : pytorch model
    opt : optimizer of the model
    xy : pytorch tensor containing inputs and labels
    """
    mets_smooth = {"loss" : None, "acc" : None}
    model.train()
    batch_size = CFG.batch_size
    batch_indices = np.array_split(np.random.permutation(len(xy)), len(xy)//batch_size + 1)
    i = 0
    mets_smooth_acc_rec = [0]
    train_end_flag = False
    for indices in tqdm(batch_indices):  # training loop
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
            # if accuracy dosent improve, stop training or reduce learning rate
            if mets_smooth["acc"] < mets_smooth_acc_rec[-1]:
                train_end_flag = opt.step_scheduler()
                if train_end_flag:
                    break
            mets_smooth_acc_rec.append(mets_smooth["acc"])            
            print("acc improve {} -> {}".format(mets_smooth_acc_rec[-2], mets_smooth_acc_rec[-1]))
    return model, train_end_flag

def inference(model, is_inference_all):
    """
    inference with a given model
    """

    print("start inference")
    model = model.eval()
    n_topk = CFG.n_topk
    if is_inference_all:
        path = CFG.val_path
        print("inference on all data")
    else:
        path = glob.glob(CFG.val_path + "/*")[0]
        print("inference on partial data")
    df_inf = pd.read_parquet(path)
    df_inf = preprocess_df(df_inf, with_label=False)
    df_inf = df_inf.groupby("session").last() # get last row of each session

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
    batch_indices = np.array_split(np.random.permutation(len(x_tensor)), len(x_tensor)//batch_size + 1)
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

        # concat inference results
        sessions_save = np.hstack([r[0] for r in recs])
        topks = np.vstack([r[1] for r in recs])
        scores = np.vstack([r[2] for r in recs])

        # save inference results
        lis = []
        dic = {}
        lis = np.zeros([len(sessions_save), n_topk*2], dtype = np.int32)
        for i in tqdm(range(len(sessions_save))):
             # first n_topk elements are topk aids, and last n_topk elements are cosine similarity score
            lis[i] = list(topks[i]) + list(scores[i])
            # mapping dict session_id to index
            dic[sessions_save[i]] = i
        arr = np.array(lis).astype(np.uint32)
        # save
        os.system(f"mkdir -p {CFG.output_path}")
        dst = CFG.output_path + CFG.output_fn.replace("emb", f"emb_{label}")
        # save matrix containing topk_aids and cosine similarities, and mapper of session to idx
        pd.to_pickle([arr, dic], dst)
        print("save! : ", dst)
        import gc; gc.collect()


class Encoder(torch.nn.Module):
    """
    nn model 
    """
    def __init__(self):
        super(Encoder, self).__init__()
        # embedding of aids
        self.emb = torch.nn.Embedding(CFG.n_aid, CFG.emb_dim, padding_idx=CFG.n_aid-1)
        # embedding of time
        self.emb_h = torch.nn.Embedding(CFG.h_max-CFG.h_min+4, CFG.time_emb_dim, padding_idx=CFG.h_max-CFG.h_min+3)

        inp_dim = CFG.emb_dim * CFG.n_length + (CFG.n_feat + 1 + CFG.time_emb_dim) * CFG.n_length
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
        
        # embedding of type
        self.emb_label_type = torch.nn.Embedding(7, CFG.emb_dim, padding_idx=0)
        
        
    def forward(self, xy_batch):
        """
        receives batch of x and y,
        returns loss and accuracy
        """

        # split data to x, y, and label_type
        x_batch = xy_batch[:,:-CFG.n_label*2]
        y_batch = xy_batch[:,-CFG.n_label*2:-CFG.n_label]
        label_type = xy_batch[:,-CFG.n_label:]

        # sampling negative aids
        neg_aids = torch.randperm(CFG.n_aid, device = xy_batch.get_device())[:CFG.n_neg_coef * len(xy_batch)]

        # calculate session embedding
        x_em = self._calc_query_emb(x_batch) # (batch_size, CFG.emb_dim)
        label_type_em = self.emb_label_type(label_type) # (batch_size, CFG.n_label, CFG.emb_dim)
        x_em = x_em.unsqueeze(1) + label_type_em # (batch_size, CFG.n_label, CFG.emb_dim)

        # calculate target aid embeddings
        y_em = self.do_emb(y_batch) # (batch_size, CFG.n_label, CFG.emb_dim)
        neg_em = self.do_emb(neg_aids) # (n_negative_sample, CFG.emb_dim)

        # normalize emebddings
        x_em = x_em/torch.norm(x_em, dim=2, keepdim = True)
        y_em = y_em/torch.norm(y_em, dim=2, keepdim = True)
        neg_em = neg_em/torch.norm(neg_em, dim=1, keepdim = True)

        # cosine similarity with positive aids
        cossim_pos_all = torch.sum(x_em * y_em, dim = 2) # (batch_size, CFG.n_label)
        y_mask = y_batch.eq(CFG.n_aid - 1)
        cossim_pos_min = torch.min(cossim_pos_all + y_mask, dim = 1)[0] # cosine similarity of most difficult positive sample
        cossim_pos_mean = torch.sum(cossim_pos_all * label_type, dim = 1) / torch.sum(label_type, dim = 1)
        cossim_pos = (cossim_pos_mean + cossim_pos_min) / 2 # (batch_size, )

        # cosine similarity with negative aids
        cossim_neg = torch.matmul(x_em[:,0], neg_em.T) # (batch_size, n_negative_sample)
        cossim_neg = torch.topk(cossim_neg, CFG.neg_topk, dim = 1)[0] # get difficult negative samples

        # calculate loss and accuracy
        cossim = torch.cat([cossim_pos.unsqueeze(1), cossim_neg], dim = 1) / CFG.temperature
        p = torch.softmax(cossim,dim=1)
        loss = - torch.mean(torch.log(p[:,0])) # first sample in each batch is positive
        acc = torch.mean((torch.max(p,dim=1)[1] == 0).to(torch.float))
        return loss, acc

        
    def _calc_query_emb(self, x_batch):
        """
        calculate session embedding
        """
        x_batch = x_batch.reshape(-1, CFG.n_feat + 1, CFG.n_length).transpose(1,2) ## (batch_size, n_length, n_feat)

        # embedding from aids
        em = self.emb(x_batch[:,:,0])

        # other features like time in a day and type of aid
        x_other = torch.cat([
            torch.sin(x_batch[:,:,1:2] % 24 / 24 * np.pi * 2),
            torch.cos(x_batch[:,:,1:2] % 24 / 24 * np.pi * 2),
            x_batch[:,:,2:],
        ], dim=2)

        # embedding from hour
        emh = self.emb_h(x_batch[:,:,1]) + \
              self.emb_h(torch.clip(x_batch[:,:,1]+1, max = CFG.h_max-CFG.h_min+3)) + \
              self.emb_h(torch.clip(x_batch[:,:,1]+2, max = CFG.h_max-CFG.h_min+3))

        # concat all embeddings
        x = torch.cat([em, x_other, emh], dim = 2)

        # masking where no history aid
        mask = ~x_batch[:,:,0].eq(CFG.n_aid-1).unsqueeze(2)
        x = x * mask
        x = x.reshape(len(x), -1) # (batch_size, CFG.n_length * sum_dim_embeddings_and_features)
        
        # mlp to create session embedding
        return self.mlp(x)  ## (batch_size, CFG.emb_dim)
    
    def calc_query_emb(self, x_batch, label_type):
        """
        calculate session embedding to target type
        calculate session embedding by self._calc_query_emb, and add target type embedding
        """
        return self._calc_query_emb(x_batch) + self.emb_label_type(label_type)
    
    def do_emb(self, x):
        return self.emb(x)
    
class MyOptimizer():
    """
    optimizer + lr scheduler
    """
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
        self.opt_em = torch.optim.AdamW(params_emb, lr = CFG.lr)
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


def load_train():
    df = pd.read_parquet(CFG.train_path)
    df = df[df["ts"] > (df["ts"].max() - (3600 * 24 * 14 * 1000))]
    if CFG.run_test:
        df = df[:10000000]
    return df

def load_test():
    df = pd.read_parquet(CFG.val_path)
    if CFG.run_test:
        df = df[:10000000]
    return df

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--pretrain", default="",
     help = "if not empty, load model")
    parser.add_argument("--is_skip_train", action='store_true',
     help = "if true, skip training and conduct only inference")
    parser.add_argument("--is_inference_all", action='store_true',
     help = "if true, inference on all data. if false inference on partial data for local validation"
    )
    args = parser.parse_args()

    if args.pretrain == "at_train":
        # load latest model 
        model_path = CFG.output_path + "/last_"+CFG.output_model_fn
    elif args.pretrain == "at_test":
        # load latest model from training directory
        model_path = "../../otto/" + CFG.output_path.replace("..", "") + "/last_"+CFG.output_model_fn
    elif args.pretrain == "":
        model_path = None

    if not args.is_skip_train:
        xy = torch.cat([preprocess_df(load_train(), with_label=True), 
                        preprocess_df(load_test(), with_label=True)], dim = 0)
        print(xy.shape)

    # create model
    if model_path is None:
        model = Encoder()
    else:
        print("load model from ", model_path)
        model = pd.read_pickle(model_path).module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

     # make parallel
    model = torch.nn.DataParallel(model)
    torch.backends.cudnn.benchmark = True
    model.do_emb = model.module.do_emb
    model.calc_query_emb = model.module.calc_query_emb
    model = model.to(device)

    if not args.is_skip_train:
        opt = MyOptimizer(model)
        os.system(f"mkdir -p {CFG.output_path}")
        print("start training")
        for i in range(CFG.n_epoch):
            model, train_end_flag = train(model, opt, xy)
            if train_end_flag:
                break
        pd.to_pickle(model.cpu(), CFG.output_path + "/" + "last"+"_"+CFG.output_model_fn)
        model = model.to(device)

    inference(model, args.is_inference_all)