import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from metrics import accuracy, kappa, macro_F1, micro_F1, median_F1, average_F1, mf1, class_report
from anndata import AnnData
import os

def eval_f1(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true, y_pred, average='micro')
        acc_list.append(f1)

    return sum(acc_list) / len(acc_list)


def eval_acc(y_true, y_pred):
    acc_list = []
    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        is_labeled = y_true[:, i] == y_true[:, i]
        correct = y_true[is_labeled, i] == y_pred[is_labeled, i]
        acc_list.append(float(np.sum(correct)) / len(correct))

    return sum(acc_list) / len(acc_list)


def eval_rocauc(y_true, y_pred):
    """ adapted from ogb
    https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/evaluate.py"""
    rocauc_list = []
    y_true = y_true.detach().cpu().numpy()
    if y_true.shape[1] == 1:
        # use the predicted class for single-class classification
        y_pred = F.softmax(y_pred, dim=-1)[:, 1].unsqueeze(1).cpu().numpy()
    else:
        y_pred = y_pred.detach().cpu().numpy()

    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
            is_labeled = y_true[:, i] == y_true[:, i]
            score = roc_auc_score(y_true[is_labeled, i], y_pred[is_labeled, i])

            rocauc_list.append(score)

    if len(rocauc_list) == 0:
        raise RuntimeError(
            'No positively labeled data available. Cannot compute ROC-AUC.')

    return sum(rocauc_list) / len(rocauc_list)

@torch.no_grad()
def get_embedding(model, dataset, split_idx, args, adata, le, save_path):
    model.eval()
    pred, layer_ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau, return_embedding=True)
    pred = pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    pred = le.inverse_transform(pred)

    embedding = layer_[-1].squeeze(0).cpu().numpy()
    adata_embedding = AnnData(embedding)
    adata_embedding.obs['CellType'] = adata.obs['CellType'].values
    adata_embedding.obs['Batch'] = adata.obs['Batch'].values
    adata_embedding.obs['Pred'] = pred
    adata_embedding.write(os.path.join(save_path, f"embedding.h5ad"))

@torch.no_grad()
def get_embedding_weight(model, dataset, split_idx, args, adata, le, save_path, train_shape, test_shape):
    model.eval()
    pred, layer_, weight_ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau, train_shape, test_shape, return_embedding=True)
    prob = pred.max(dim=-1).values.detach().cpu().numpy()
    pred = pred.argmax(dim=-1, keepdim=True).detach().cpu().numpy()
    pred = le.inverse_transform(pred)

    embedding = layer_[-1].squeeze(0).cpu().numpy()
    adata_embedding = AnnData(embedding)
    adata_embedding.obs['CellType'] = adata.obs['CellType'].values
    adata_embedding.obs['Batch'] = adata.obs['Batch'].values
    adata_embedding.obs['Pred'] = pred
    adata_embedding.obs['Prob'] = prob
    adata_embedding.obs.index = adata.obs.index.values
    adata_embedding.write(os.path.join(save_path, f"embedding.h5ad"))
    
    np.save(os.path.join(save_path, f"weight.npy"), weight_)


@torch.no_grad()
def evaluate(model, dataset, split_idx, eval_func, criterion, args, le):
    model.eval()
    out, _ = model(dataset.graph['node_feat'], dataset.graph['adjs'], args.tau)

    train_acc = eval_func(
        dataset.label[split_idx['train']], out[split_idx['train']])
    valid_acc = eval_func(
        dataset.label[split_idx['valid']], out[split_idx['valid']])
    # test_acc = eval_func(
    #     dataset.label[split_idx['test']], out[split_idx['test']])

    
    # t_acc = accuracy(
    #     out[split_idx['test']], dataset.label[split_idx['test']])
    # t_kappa = kappa(
    #     out[split_idx['test']], dataset.label[split_idx['test']])
    # t_macro_F1 = macro_F1(
    #     out[split_idx['test']], dataset.label[split_idx['test']])
    # t_micro_F1 = micro_F1(
    #     out[split_idx['test']], dataset.label[split_idx['test']])
    # t_median_F1 = median_F1(
    #     out[split_idx['test']], dataset.label[split_idx['test']])
    # t_average_F1 = average_F1(
    #     out[split_idx['test']], dataset.label[split_idx['test']])
    # t_mF1 = mf1(
    #     out[split_idx['test']], dataset.label[split_idx['test']])
    # t_class_report = class_report(
    #     out[split_idx['test']], dataset.label[split_idx['test']], le)
    
    # train_class_reprot = class_report(
    #     out[split_idx['train']], dataset.label[split_idx['train']], le)

    out = F.log_softmax(out, dim=1)
    valid_loss = criterion(
        out[split_idx['valid']], dataset.label.squeeze(1)[split_idx['valid']])

    return train_acc, valid_acc, valid_loss
