import warnings
warnings.filterwarnings("ignore")
import argparse
import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import remove_self_loops, add_self_loops

from logger import Logger
from dataset import load_ATAC_dataset
from data_utils import load_fixed_splits, adj_mul
from eval import evaluate, eval_acc, eval_rocauc, eval_f1, get_embedding, get_embedding_weight
from parse import parser_add_main_args
import pandas as pd
from tqdm import tqdm

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

# get args
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()

# save args
if not os.path.exists(args.save_path):
    os.mkdir(args.save_path)
save_path = os.path.join(args.save_path, args.save_name)
if not os.path.exists(save_path):
    os.mkdir(save_path)
f = open(os.path.join(save_path, "args.txt"), "w")
f.write('Args:\n')
for k, v in sorted(vars(args).items()):
    f.write('\t{}: {}\n'.format(k, v))

# fix seed
fix_seed(args.seed)

# set device
if args.cpu:
    device = torch.device("cpu")
else:
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

# load and preprocess data
dataset, adata, le, train_shape, test_shape = load_ATAC_dataset(args.data_dir, args.train_name_list, args.test_name, args.sample_ratio, args.edge_ratio, save_path, args.save_unknown, args.save_rare, args.no_smote)

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

# get the splits of train, valid and test
split_idx_lst = load_fixed_splits(dataset)

# Basic information of datasets
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

# select model based on whether weight is needed
if args.get_weight:
    from model_weight import GraphTransformer
else:
    from model import GraphTransformer

# make model
model=GraphTransformer(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                    num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                    use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
                    nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)

criterion = nn.NLLLoss()

# select performance metric
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

# make logger
logger = Logger(args.runs, args)

model.train()

# Adj storage for relational bias
adjs = []
adj, _ = remove_self_loops(dataset.graph['edge_index'])
adj, _ = add_self_loops(adj, num_nodes=n)
adjs.append(adj)
for i in range(args.rb_order - 1):
    adj = adj_mul(adj, adj, n)
    adjs.append(adj)
dataset.graph['adjs'] = adjs

# Training loop
for run in range(args.runs):
    split_idx = split_idx_lst[0]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('inf')
    class_report = None

    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()

        out, link_loss_ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)

        out = F.log_softmax(out, dim=1)
        loss = criterion(
            out[train_idx], dataset.label.squeeze(1)[train_idx])
        
        loss -= args.lamda * sum(link_loss_) / len(link_loss_)
        loss.backward()
        optimizer.step()

        if epoch % args.eval_step == 0:
            # compute performance metric
            result = evaluate(model, dataset, split_idx, eval_func, criterion, args, le)
            logger.add_result(run, result)

            if result[3] < best_val:
                # save the best model on the validation set
                best_val = result[3]
                torch.save(model.state_dict(), os.path.join(save_path, "model.pkl"))

    # print results
    result = logger.print_statistics(run, mode=None)
    dict_result = {
        "acc": result[4],
        "kappa": result[5],
        "macro F1": result[6],
        "micro F1": result[7],
        "median F1": result[8],
        "average F1": result[9],
        "mF1": result[10],
    }
    df = pd.DataFrame(dict_result, index=[0])
    df.to_csv(os.path.join(save_path, "result.csv"))

    # get embedding of data
    best_val_model = GraphTransformer(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                    num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                    use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
                    nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)
    best_val_model.load_state_dict(torch.load(os.path.join(save_path, "model.pkl")))

    # save weight if they are needed
    if args.get_weight:
        get_embedding_weight(best_val_model, dataset, split_idx, args, adata, le, save_path, train_shape, test_shape)
    else:
        get_embedding(best_val_model, dataset, split_idx, args, adata, le, save_path)