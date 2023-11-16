# This part of the code is referenced from scATAnno.
import numpy as np
import pandas as pd
import os
import scanpy as sc
from anndata import AnnData
from typing import Optional, List, Union
import anndata as ad
from anndata.experimental import AnnCollection
import math

def get_sigmas(dists):
    sigma = []
    for row in dists:
        s = (sum( x**2.0 for x in row ) / float(len(row)) )**0.5
        sigma.append(s)
    return sigma

def gaussian_kernal(dists, sigma):
    # gaussian kernel function to convert distance to similarity scores
    sigma_n = np.array(sigma)[:, np.newaxis]
    K = np.exp(-dists / ( (2 / sigma_n) **2))
    return K

# Assign celltype labels without filtering by uncertainty score
def raw_assignment(K, neighbors_labels):
    prediction = []
    uncertainty = []
    for i in range(K.shape[0]):
        c_label = neighbors_labels[i,:]
        c_D = K[i,:]
        c_df = pd.DataFrame({'label': c_label,'dist': c_D})
        p = c_df.groupby('label').sum('dist')/np.sum(c_df['dist'])
        u = 1 - p
        pred_y = u.index[np.argmin(u)]
        prediction.append(pred_y)
        uncertainty.append(np.min(u).values[0])
    return prediction, uncertainty



def load_reference_data(path):
    """read reference atlas
    Parameters
    ----------
    path: path to reference h5ad data 
    Returns reference anndata
    -------
    If h5ad file not found, search for MTX and TSV files; if none found, raise error
    """
    parent_path = os.path.dirname(os.path.normpath(path))
    if os.path.isfile(path):
        try:
            reference_data = sc.read_h5ad(path)
            reference_data.obs['dataset'] = "reference"
            return reference_data
        except OSError as error:
            print("refernce anndata not found")
            pass
    elif os.path.isfile(os.path.join(parent_path, "atac_atlas.mtx")) & os.path.isfile(os.path.join(parent_path, "atac_atlas_genes.tsv")) & os.path.isfile(os.path.join(parent_path, "atac_atlas_cellbarcodes.tsv")):
        reference_data = convert_mtx2anndata_simple(path, mtx_file = "atac_atlas.mtx",cells_file = "atac_atlas_cellbarcodes.tsv",features_file = "atac_atlas_genes.tsv")
        return reference_data
    else: raise FileNotFoundError

def import_query_data(path, mtx_file,cells_file,features_file, variable_prefix, celltype_col="celltypes", add_metrics = True):
    """convert the count matrix into an anndata.
    Parameters
    ----------
    path: data directory including mtx, barcodes and features
    mtx_file: mtx filename 
    cells_file: cell barcode filename
    features_file: feature filename
    variable_prefix: sample name prefix
    celltype_col: column name of cell types, default is "celltypes"
    add_metrics: whether adding metadata of metrics from QuickATAC
    
    Returns a AnnData object
    -------
    """
    # create anndata
    data = sc.read_mtx(os.path.join(path,mtx_file))
    data = data.T
    features = pd.read_csv(os.path.join(path, features_file), header=None, sep= '\t')
    barcodes = pd.read_csv(os.path.join(path, cells_file), header=None)

    # Split feature matrix and set peak separated by (:, -) to match reference peaks
    data.var_names = features[0]
    data.obs_names = barcodes[0]
    
    data.obs[celltype_col] = variable_prefix
    data.obs['tissue'] = variable_prefix
    data.obs['dataset'] = variable_prefix
    
    # remove spike-in cell
    data = data[data.obs.index != "spike-in"]
    # add qc filtering metrics from quickATAC if add_metrics set to true
    if add_metrics == True:
        import glob
        try:
            metrics_filepath = glob.glob(os.path.join(path, "*meta*"))[0]
            metrics = pd.read_csv(metrics_filepath, sep='\t', index_col=0)
            metrics = metrics[metrics.index != "spike-in"]
            data.obs = pd.merge(data.obs, metrics, right_index=True, left_index = True)
        except OSError as error:
            import warnings
            warnings.warn('Metrics file not found, anndata returned with no meta metrics')
            return data
    return data




def umap(
    adata: AnnData,
    n_comps: int = 2,
    use_dims: Optional[Union[int, List[int]]] = None,
    use_rep: Optional[str] = None,
    key_added: str = 'umap',
    random_state: int = 0,
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """
    Parameters
    ----------
    data
        AnnData.
    n_comps
        The number of dimensions of the embedding.
    use_dims
        Use these dimensions in `use_rep`.
    use_rep
        Use the indicated representation in `.obsm`.
    key_added
        `adata.obs` key under which to add the cluster labels.
    random_state
        Random seed.
    inplace
        Whether to store the result in the anndata object.

    Returns
    -------
    None
    """
    from umap import UMAP

    if use_rep is None: use_rep = "X_spectral"
    if use_dims is None:
        data = adata.obsm[use_rep]
    elif isinstance(use_dims, int):
        data = adata.obsm[use_rep][:, :use_dims]
    else:
        data = adata.obsm[use_rep][:, use_dims]
    # newly added
    data = np.asarray(data)
    umap = UMAP(
        random_state=random_state, n_components=n_comps
        ).fit_transform(data)
    if inplace:
        adata.obsm["X_" + key_added] = umap
    else:
        return umap



def get_umap(integrated_adata, out_dir, use_rep = "X_spectral_harmony", save = True, filename='1.Merged_query_reference.h5ad'):
    """
    Plot UMAP for integrated scATAC-seq data
    Parameters
    ----------
    integrated_adata
        AnnData.
    out_dir
        Directory to save adata
    use_rep
        Use the indicated representation in `.obsm`.
    save
        Whether to save the anndata object and spectral embeddings
    filename: the filename of stored anndata object and spectral embeddings

    Returns
    -------
    None
    """
    if use_rep in integrated_adata.obsm:
        umap(integrated_adata, use_rep=use_rep)
    else: raise ValueError("Missing low dimensionality")
    # Save AnnData object
    if save == True:
        tmp_adata = integrated_adata.copy()
        if 'X_spectral' in tmp_adata.obsm: 
            tmp = pd.DataFrame(tmp_adata.obsm['X_spectral'])
            tmp.index= tmp_adata.obs.index
            tmp.to_csv(os.path.join(out_dir,'X_spectral.csv'))
            del tmp_adata.obsm['X_spectral']
        if 'X_spectral_harmony' in tmp_adata.obsm: 
            tmp = pd.DataFrame(tmp_adata.obsm['X_spectral_harmony'])
            tmp.index= tmp_adata.obs.index
            tmp.to_csv(os.path.join(out_dir,'X_spectral_harmony.csv'))
            del tmp_adata.obsm['X_spectral_harmony']
        tmp_adata.write(os.path.join(out_dir,filename))
    return(integrated_adata)

def cluster_annotation_anndata(adata, prediction_col = None, cluster_col = None):
    if cluster_col is None:
        cluster_col = "Clusters"
    else: cluster_col = cluster_col
    
    if prediction_col is None:
        prediction_col = "corrected_pred_y_major"
    else: prediction_col = prediction_col
    
    if cluster_col in adata.obs.columns:
        cluster_anno_unstack = adata.obs.groupby(cluster_col)[prediction_col].value_counts().unstack()
    else: raise KeyError("Column {} Not Found in dataframe".format(cluster_col))
    
    cluster_group_anno = {}
    for i in cluster_anno_unstack.index:
        cluster_group_anno[i] = cluster_anno_unstack.columns[np.argmax(cluster_anno_unstack.loc[i,:])]
    
    cluster_annotations = []
    for cell_idx in range(adata.obs.shape[0]):
        key = adata.obs.iloc[cell_idx, :][cluster_col]
        anno = cluster_group_anno[key]
        cluster_annotations.append(anno)
    adata.obs['cluster_annotation'] = cluster_annotations
    return adata

def cluster_assign(query, use_rep, cluster_col=None, UMAP=True, leiden_resolution=3):
    """
    Return query data with cluster-level annotation

    Parameters
    ----------
    query: anndata of query cells
    cluster_col: if None, automatically cluster by leiden algorithm; otherwise, leiden cluster and then input cluster column name
    UMAP: if True, redo UMAP for query data; else, do not change UMAP
    """
    query_only_newUMAP = query.copy()
    if UMAP:
        sc.pp.neighbors(query_only_newUMAP, use_rep=use_rep)
        sc.tl.umap(query_only_newUMAP)
    if cluster_col is None:
        sc.tl.leiden(query_only_newUMAP,  key_added = "leiden", resolution = leiden_resolution)
        # query_only_newUMAP = cluster_annotation_anndata(query_only_newUMAP,  cluster_col = "leiden", prediction_col = "2.corrected_celltype")
        query_only_newUMAP = cluster_annotation_anndata(query_only_newUMAP,  cluster_col = "leiden", prediction_col = "pred_y_unknown")
    else:
        # query_only_newUMAP = cluster_annotation_anndata(query_only_newUMAP,  cluster_col = cluster_col, prediction_col = "2.corrected_celltype")
        query_only_newUMAP = cluster_annotation_anndata(query_only_newUMAP,  cluster_col = cluster_col, prediction_col = "pred_y_unknown")

    return query_only_newUMAP


def select_features(
    adata: Union[ad.AnnData, AnnCollection],
    variable_feature: bool = True,
    whitelist: Optional[str] = None,
    blacklist: Optional[str] = None,
    inplace: bool = True,
) -> Optional[np.ndarray]:
    """
    Perform feature selection.

    Parameters
    ----------
    adata
        AnnData object
    variable_feature
        Whether to perform feature selection using most variable features
    whitelist
        A user provided bed file containing genome-wide whitelist regions.
        Features that are overlapped with these regions will be retained.
    blacklist 
        A user provided bed file containing genome-wide blacklist regions.
        Features that are overlapped with these regions will be removed.
    inplace
        Perform computation inplace or return result.
    
    Returns
    -------
    Boolean index mask that does filtering. True means that the cell is kept.
    False means the cell is removed.
    """
    if isinstance(adata, ad.AnnData):
        count = np.ravel(adata.X[...].sum(axis = 0))
    else:
        count = np.zeros(adata.shape[1])
        for batch, _ in adata.iterate_axis(5000):
            count += np.ravel(batch.X[...].sum(axis = 0))

    selected_features = count != 0

    if whitelist is not None:
        selected_features &= internal.intersect_bed(list(adata.var_names), whitelist)
    if blacklist is not None:
        selected_features &= not internal.intersect_bed(list(adata.var_names), blacklist)

    if variable_feature:
        mean = count[selected_features].mean()
        std = math.sqrt(count[selected_features].var())
        selected_features &= np.absolute((count - mean) / std) < 1.65

    if inplace:
        adata.var["selected"] = selected_features
    else:
        return selected_features

def curate_celltype_names(l, atlas):
    """
    Return a list with curated cell type based on the reference atlas
    """
    if atlas == "PBMC":
        l_new = [ 'Naive CD4 T' if i == 'Naive Treg' else i for i in l]
        l_new = [ 'NK' if i == 'Mature NK' or (i=='Immature NK') else i for i in l_new]

    elif atlas == "HealthyAdult":
        curated_major = []
        for i in l:
            if i == "B Lymphocyte":
                curated_major.append("Immune Cells")
            elif i == "T Lymphocyte":
                curated_major.append("Immune Cells")
            elif i == 'Myeloid / Macrophage':
                curated_major.append("Immune Cells")
            else: curated_major.append(i)
        l_new = curated_major
        
    elif atlas == "TIL":
        l_new = l
        #todo: merge NK1 and NK2
    else:
        l_new = l 
    return l_new


def make_anndata(adata, chrom, start, end, path):
    adata.var['chr'] = chrom
    adata.var['start'] = start
    adata.var['end'] = end
    
    sc.pp.filter_cells(adata, min_genes=0)
    sc.pp.filter_genes(adata, min_cells=0)
    
    thres = int(adata.shape[0]*0.01)
    adata = adata[:, adata.var['n_cells']>thres]

    chrs = ['chr'+str(i) for i in range(1,23)] + ['chrX', 'chrY']
    adata = adata[:, adata.var['chr'].isin(chrs)]
    
    # print(adata)
    adata.write(path)
    return adata

def get_uncertainty_score_step1(query, reference, weight_path, n_neighbors=30):
    
    # load weight
    weight = np.load(weight_path)
    
    # top k weight
    indices = np.argsort(weight)[:, -n_neighbors:]
    
    # compute distance
    dists = np.take_along_axis(weight, indices, axis=1)
    dists = 1 - dists
    
    # get labels of neighbors
    labels = np.array(reference.obs["celltypes"].values)
    neighbors_labels = []
    for i in range(indices.shape[0]):
        neighbor_label = []
        for j in range(indices.shape[1]):
            neighbor_label.append(labels[indices[i,j]])
        neighbors_labels.append(neighbor_label)
    neighbors_labels = np.array(neighbors_labels)
    
    # compute uncertainty_score
    sigma = get_sigmas(dists)
    K = gaussian_kernal(dists, sigma)
    pred_res_major = raw_assignment(K, neighbors_labels)
    
    query.obsm["kernel_distance"] = K
    query.obsm["distance"] = dists
    query.obsm["indices"] = indices
    query.obsm["neighbors_labels"] = neighbors_labels
    pred_label_major = pred_res_major[0]
    query.obs["uncertainty_score_step1"] = pred_res_major[1]
    query.obs["pred_y"] = pred_label_major
    
    return query