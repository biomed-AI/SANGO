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
from eval import evaluate, eval_acc, eval_rocauc, eval_f1, get_embedding
from parse import parser_add_main_args

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from model import GraphTransformer

# NOTE: for consistent data splits, see data_utils.rand_train_test_idx
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

### Parse args ###
parser = argparse.ArgumentParser(description='General Training Pipeline')
parser_add_main_args(parser)
args = parser.parse_args()
print(args)

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

### Load and preprocess data ###
dataset, adata, le = load_ATAC_dataset(args.data_dir, args.train_name_list, args.test_name, args.sample_ratio, args.edge_ratio, save_path)

# # save input
# adata.write(os.path.join(save_path, "raw.h5ad"))

if len(dataset.label.shape) == 1:
    dataset.label = dataset.label.unsqueeze(1)
dataset.label = dataset.label.to(device)

# get the splits for all runs
split_idx_lst = load_fixed_splits(dataset)

### Basic information of datasets ###
n = dataset.graph['num_nodes']
e = dataset.graph['edge_index'].shape[1]
# infer the number of classes for non one-hot and one-hot labels
c = max(dataset.label.max().item() + 1, dataset.label.shape[1])
d = dataset.graph['node_feat'].shape[1]

print(f"num nodes {n} | num edge {e} | num node feats {d} | num classes {c}")

print(f"train {split_idx_lst[0]['train'].shape} | valid {split_idx_lst[0]['valid'].shape} | test {split_idx_lst[0]['test'].shape}")

dataset.graph['edge_index'], dataset.graph['node_feat'] = \
    dataset.graph['edge_index'].to(device), dataset.graph['node_feat'].to(device)

### Load method ###
model=GraphTransformer(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                    num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                    use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
                    nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)

criterion = nn.NLLLoss()

### Performance metric (Acc, AUC, F1) ###
if args.metric == 'rocauc':
    eval_func = eval_rocauc
elif args.metric == 'f1':
    eval_func = eval_f1
else:
    eval_func = eval_acc

logger = Logger(args.runs, args)

model.train()
# print('MODEL:', model)

### Adj storage for relational bias ###
adjs = []
adj, _ = remove_self_loops(dataset.graph['edge_index'])
adj, _ = add_self_loops(adj, num_nodes=n)
adjs.append(adj)
for i in range(args.rb_order - 1):
    adj = adj_mul(adj, adj, n)
    adjs.append(adj)
dataset.graph['adjs'] = adjs

### Training loop ###
for run in range(args.runs):
    split_idx = split_idx_lst[0]
    train_idx = split_idx['train'].to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=args.weight_decay, lr=args.lr)
    best_val = float('inf')
    class_report = None

    for epoch in range(args.epochs):
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
            result = evaluate(model, dataset, split_idx, eval_func, criterion, args, le)
            logger.add_result(run, result[:11])

            if result[3] < best_val:
                best_val = result[3]
                test_class_report = result[11]
                train_class_report = result[12]
                torch.save(model.state_dict(), os.path.join(save_path, "model.pkl"))

            print(f'Epoch: {epoch:02d}, '
                  f'Loss: {loss:.4f}, '
                  f'Valid Loss: {result[3]:.4f}, '
                  f'Train: {100 * result[0]:.2f}%, '
                  f'Valid: {100 * result[1]:.2f}%, '
                  f'Test: {100 * result[2]:.2f}%, '
                  f'acc: {result[4]:.4f}, '
                  f'kappa: {result[5]:.4f}, '
                  f'macro_F1: {result[6]:.4f}, '
                  f'micro_F1: {result[7]:.4f}, '
                  f'median_F1: {result[8]:.4f}, '
                  f'average_F1: {result[9]:.4f}, '
                  f'mF1: {result[10]:.4f}, '
                  )

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

    # df_test_class_report = pd.DataFrame(test_class_report).T
    # df_test_class_report.to_csv(os.path.join(save_path, "test_class_report.csv"))
    # df_train_class_report = pd.DataFrame(train_class_report).T
    # df_train_class_report.to_csv(os.path.join(save_path, "train_class_report.csv"))


    best_val_model = GraphTransformer(d, args.hidden_channels, c, num_layers=args.num_layers, dropout=args.dropout,
                    num_heads=args.num_heads, use_bn=args.use_bn, nb_random_features=args.M,
                    use_gumbel=args.use_gumbel, use_residual=args.use_residual, use_act=args.use_act, use_jk=args.use_jk,
                    nb_gumbel_sample=args.K, rb_order=args.rb_order, rb_trans=args.rb_trans).to(device)
    best_val_model.load_state_dict(torch.load(os.path.join(save_path, "model.pkl")))

    get_embedding(best_val_model, dataset, split_idx, args, adata, le, save_path)