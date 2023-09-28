import numpy as np
import torch
from data_utils import rand_train_test_idx, class_rand_splits
from sklearn import preprocessing
import os
import pandas as pd
import scanpy as sc
from anndata import AnnData
import bbknn
from collections import Counter
from imblearn.over_sampling import SMOTE


class NCDataset(object):
    def __init__(self, name):
        """
        based off of ogb NodePropPredDataset
        https://github.com/snap-stanford/ogb/blob/master/ogb/nodeproppred/dataset.py
        Gives torch tensors instead of numpy arrays
            - name (str): name of the dataset
            - root (str): root directory to store the dataset folder
            - meta_dict: dictionary that stores all the meta-information about data. Default is None, 
                    but when something is passed, it uses its information. Useful for debugging for external contributers.

        Usage after construction:

        split_idx = dataset.get_idx_split()
        train_idx, valid_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph, label = dataset[0]

        Where the graph is a dictionary of the following form:
        dataset.graph = {'edge_index': edge_index,
                         'edge_feat': None,
                         'node_feat': node_feat,
                         'num_nodes': num_nodes}
        For additional documentation, see OGB Library-Agnostic Loader https://ogb.stanford.edu/docs/nodeprop/

        """

        self.name = name  # original name, e.g., ogbn-proteins
        self.graph = {}
        self.label = None

    def get_idx_split(self, split_type='random', train_prop=.5, valid_prop=.25, label_num_per_class=20):
        """
        split_type: 'random' for random splitting, 'class' for splitting with equal node num per class
        train_prop: The proportion of dataset for train split. Between 0 and 1.
        valid_prop: The proportion of dataset for validation split. Between 0 and 1.
        label_num_per_class: num of nodes per class
        """

        if split_type == 'random':
            ignore_negative = False if self.name == 'ogbn-proteins' else True
            train_idx, valid_idx, test_idx = rand_train_test_idx(
                self.label, train_prop=train_prop, valid_prop=valid_prop, ignore_negative=ignore_negative)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        elif split_type == 'class':
            train_idx, valid_idx, test_idx = class_rand_splits(self.label, label_num_per_class=label_num_per_class)
            split_idx = {'train': train_idx,
                         'valid': valid_idx,
                         'test': test_idx}
        return split_idx

    def __getitem__(self, idx):
        assert idx == 0, 'This dataset has only one graph'
        return self.graph, self.label

    def __len__(self):
        return 1

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__, len(self))

def check_graph_error_rate(datastr, batch_info, train_label, test_label, save_path):
    error_count=0
    batch_num=len(batch_info)
    batch=np.concatenate([[i]*batch_info[i] for i in range(batch_num)],axis=0)

    label1 = train_label
    label2 = test_label

    labels=np.concatenate((label1, label2), axis=0)
    labels = pd.DataFrame(labels)
    label_types=np.unique(labels).tolist()
    rename = {label_types[i]:i for i in range(len(label_types))}
    labels = labels.replace(rename).values.flatten()
    f=lambda x:(x*(x+1)//2)
    batch_error_count=[0 for j in range(f(batch_num))]
    batch_count=[0 for j in range(f(batch_num))]
    graph_path=os.path.join(save_path, "edge.txt")
    data=pd.read_table(graph_path).values
    total_count = data.shape[0]
    for value in data:
        edge_con=True
        split_value= value[0].split(' ')
        index1, index2=int(split_value[0]), int(split_value[1])
        x=batch[index1]
        y=batch[index2]
        x,y=max(x,y),min(x,y)
        if labels[index1]!=labels[index2]:
            error_count+=1
            edge_con=False
        batch_count[f(x)+y]+=1
        if not edge_con:
            batch_error_count[f(x)+y]+=1
    file=open(os.path.join(save_path, "edge_error.txt"),'w+')
    file.write("graph error: %.4f\n"%(error_count/total_count))
    for i in range(batch_num):
        for j in range(i+1):
            file.write('The edge count between batch%d and batch%d is %d\n'%(i+1,j+1,batch_count[f(i)+j]))
            if batch_count[f(i)+j] > 0:
                file.write('The edge error rate between batch%d and batch%d is %.4f\n'%(i+1,j+1,batch_error_count[f(i)+j]/batch_count[f(i)+j]))
    file.close()

def bbknn_construct_graph(datastr_name, data, batch_info,edge_ratio, train_label, test_label, save_path):
    sc.settings.verbosity = 3
    batch_num=len(batch_info)
    batch_labels=np.concatenate([['batch%d'%(i+1)]*batch_info[i] for i in range(batch_num)],axis=0)
    formatting = AnnData(data)
    formatting.obs["batch"] = batch_labels
    adata=formatting
    sc.tl.pca(adata)
    bbknn.bbknn(adata)

    sc.tl.leiden(adata, resolution=0.4) 
    # resolution: A parameter value controlling the coarseness of the clustering.
    # Higher values lead to more clusters
    bbknn.ridge_regression(adata, batch_key=['batch'], confounder_key=['leiden'])
    sc.pp.pca(adata)
    bbknn.bbknn(adata, batch_key='batch') 

    cell_count=adata.obsp['connectivities'].shape[0]
    graph_mtx=adata.obsp['connectivities'].tocoo()
    rows=graph_mtx.row
    cols=graph_mtx.col
    ratio_value = graph_mtx.data
    batch=np.concatenate([[i]*batch_info[i] for i in range(batch_num)],axis=0)

    inter_ratio=[[[] for i in range(batch_num)] for j in range(batch_num)]
    inter_num=0
    outer_num=0
    for i in range(len(rows)):
        if batch[rows[i]]!=batch[cols[i]]:
            x=batch[rows[i]]
            y=batch[cols[i]]
            x,y=max(x,y),min(x,y)
            inter_ratio[x][y].append(ratio_value[i])

    times=edge_ratio
    for i in range(batch_num):
        for j in range(batch_num):
            if len(inter_ratio[i][j])>0:
                inter_ratio[i][j].sort(reverse=True)

    f = open(os.path.join(save_path, "edge.txt"), 'w')
    print("total edges: ", len(inter_ratio[1][0]))
    print("edge_ratio: ", edge_ratio) 
    for i in range(len(rows)):
        if batch[rows[i]]==batch[cols[i]]:
            f.write('{} {}\n'.format(rows[i], cols[i]))
        else:
            x=batch[rows[i]]
            y=batch[cols[i]]
            x,y=max(x,y),min(x,y)
            ratio=inter_ratio[x][y][min(int(times*max(batch_info[x], batch_info[y])), len(inter_ratio[x][y])-1)]
            if ratio_value[i] > ratio:  # trim the edges between batches
                f.write('{} {}\n'.format(rows[i], cols[i]))

    for index in range(cell_count):
        f.write('{} {}\n'.format(index, index))
    f.close()
    check_graph_error_rate(datastr_name,batch_info,train_label, test_label, save_path)

def atac_data_graph_construction(data_path, train_name_list, test_name, sample_ratio, edge_ratio, label_name, save_path):
    # load data
    adata = sc.read_h5ad(data_path)

    # train data
    adata_train = None
    batch = []
    print("shape of train data:")
    for train_name in train_name_list:
        adata_iter = adata[adata.obs['Batch'] == train_name]
        print(f"\t{adata_iter.shape}")
        if adata_train is None:
            adata_train = adata_iter
        else:
            adata_train = sc.AnnData.concatenate(adata_train, adata_iter)
        batch += [train_name] * adata_iter.shape[0]
    adata_train.obs['batch'] = batch
    print("shape of concat train data:")
    print(f"\t{adata_train.shape}")

    adata_test = None
    print("shape of test data:")
    for test_name_iter in test_name:
        adata_iter = adata[adata.obs["Batch"] == test_name_iter]
        print(f"\t{adata_iter.shape}")
        if adata_test is None:
            adata_test = adata_iter
        else:
            adata_test = sc.AnnData.concatenate(adata_test, adata_iter)
    print("shape of concat test data:")
    print(f"\t{adata_test.shape}")

    #----- preprocess -----

    # remove unknown
    adata_train = adata_train[adata_train.obs[label_name] != 'Unknown']
    adata_train = adata_train[adata_train.obs[label_name] != 'unknown']
    #adata_train = adata_train[adata_train.obs[label_name].values.notnull()]

    adata_test = adata_test[adata_test.obs[label_name] != 'Unknown']
    adata_test = adata_test[adata_test.obs[label_name] != 'unknown']
    #adata_test = adata_test[adata_test.obs[label_name].values.notnull()]

    print(f"----------after remove unknown----------")
    print(f"shape of train data: {adata_train.shape}")
    print(f"shape of test data: {adata_test.shape}")

    # remove rare
    class_list = np.unique(adata_train.obs[label_name].values)
    for i in class_list:
        if sum(adata_train.obs[label_name].values == i) <= 10:
            adata_train = adata_train[adata_train.obs[label_name] != i]
    
    class_list = np.unique(adata_test.obs[label_name].values)
    for i in class_list:
        if sum(adata_test.obs[label_name].values == i) <= 10:
            adata_test = adata_test[adata_test.obs[label_name] != i]
    
    print(f"----------after remove rare----------")
    print(f"shape of train data: {adata_train.shape}")
    print(f"shape of test data: {adata_test.shape}")
                
    # intersection
    intersection = np.intersect1d(
        np.unique(adata_train.obs[label_name].values), 
        np.unique(adata_test.obs[label_name].values)
    )
    difference = np.setdiff1d(np.unique(adata_test.obs[label_name].values), intersection)
    for i in difference:
        adata_test = adata_test[adata_test.obs[label_name] != i]
    
    print(f"----------after intersect----------")
    print(f"shape of train data: {adata_train.shape}")
    print(f"shape of test data: {adata_test.shape}")

    # resample
    x = adata_train.X
    y = adata_train.obs[label_name].values
    print('Original dataset shape %s' % Counter(y))

    class_num_dict = {}
    min_num = (int)(adata_train.shape[0] * sample_ratio)
    print(min_num)
    for i in np.unique(adata_train.obs[label_name].values):
        class_num_dict[i] = max(sum(adata_train.obs[label_name] == i), min_num)

    sm = SMOTE(sampling_strategy=class_num_dict)
    x_res, y_res = sm.fit_resample(x, y)
    print('Resampled dataset shape %s' % Counter(y_res))

    adata_train = AnnData(x_res)
    adata_train.obs[label_name] = y_res

    data = np.vstack((adata_train.X, adata_test.X))
    batch_info = []
    # for train_name in train_name_list:
    #    batch_info.append(adata_train[adata_train.obs['batch'] == train_name].shape[0])
    batch_info.append(adata_train.shape[0])
    batch_info.append(adata_test.shape[0])

    datastr_name = ""
    for train_name in train_name_list:
        datastr_name += train_name + "_"
    for test_name_iter in test_name:
        datastr_name += test_name_iter + "_"

    bbknn_construct_graph(datastr_name, data, batch_info, edge_ratio, adata_train.obs[label_name].values, adata_test.obs[label_name].values, save_path)

    return adata_train, adata_test

def ATAC_Dataset(data_path, train_name_list, test_name, sample_ratio, edge_ratio, label_name, save_path, le):
        # load data
        adata_train, adata_test = atac_data_graph_construction(data_path, train_name_list, test_name, sample_ratio, edge_ratio, label_name, save_path)
        edge = pd.read_csv(os.path.join(save_path, "edge.txt"), header=None, sep=" ")
        # data and label
        adata = sc.AnnData.concatenate(adata_train, adata_test)
        labels = adata.obs[label_name].values
        labels = le.fit_transform(labels)
        n_classes = max(labels)+1
        print(le.classes_)
        print(f"n_classes: {n_classes}")


        edge_index = torch.from_numpy(edge.to_numpy().T)
        node_feat = torch.from_numpy(adata.X)
        label = torch.from_numpy(labels)
        num_nodes = adata.shape[0]

        # split train and valid
        val_ratio = 0.2
        train_mask = []
        val_mask = []
        for i in range(n_classes):
            idx = np.array(np.where(labels == i)).squeeze()
            # only train
            idx = idx[idx < adata_train.shape[0]]
            # shuffle
            np.random.shuffle(idx)
            # split
            num = idx.shape[0]
            val_num = (int)(num * val_ratio)
            val_mask.extend(idx[:val_num])
            train_mask.extend(idx[val_num:])

        train_mask = torch.from_numpy(np.array(train_mask))
        val_mask = torch.from_numpy(np.array(val_mask))
        test_mask = torch.arange(adata_train.shape[0], adata.shape[0])

        return edge_index, node_feat, label, num_nodes, train_mask, val_mask, test_mask, adata

def load_ATAC_dataset(data_dir, train_name, test_name, sample_ratio, edge_ratio, save_path, label_name='CellType'):
    le = preprocessing.LabelEncoder()
    edge_index, node_feat, label, num_nodes, train_mask, val_mask, test_mask, adata = ATAC_Dataset(data_dir, train_name, test_name, sample_ratio, edge_ratio, label_name, save_path, le)

    dataset = NCDataset('ATAC')

    dataset.train_idx = train_mask
    dataset.valid_idx = val_mask
    dataset.test_idx = test_mask

    dataset.graph = {'edge_index': edge_index,
                     'node_feat': node_feat,
                     'edge_feat': None,
                     'num_nodes': num_nodes}
    dataset.label = label

    return dataset, adata, le



