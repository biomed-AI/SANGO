import numpy as np
import h5py
from torch.utils.data import Dataset
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix
import logging
logger = logging.getLogger(__name__)

def load_adata(data) -> AnnData:
    '''
    load data as AnnData

    data: path to the input h5ad file

    return: AnnData
    '''
    adata = sc.read_h5ad(data)
    if adata.X.max() > 1:
        logger.info("binarized")
        adata.X.data = (adata.X.data > 0).astype(np.float32)
    return adata

class SingleCellDataset(Dataset):
    '''
    preprocess data and make dataset

    data: AnnData
    genome: reference genome
    seq_len: length to extend/trim sequences to

    return: dataset
    '''
    def __init__(self, data: AnnData, genome, seq_len=1344):
        # gene need to be accessible in 1% cells
        sc.pp.filter_genes(data, min_cells=int(round(0.01 * data.shape[0])))
        self.data = data
        self.seq_len = seq_len
        # load genome
        self.genome = h5py.File(genome, 'r')
        self.obs = self.data.obs.copy()
        del self.data.obs
        self.var = self.data.var.copy()
        del self.data.var
        self.X = csr_matrix(self.data.X.T)
        del self.data.X

        if "chr" in self.var.keys():
            self.chroms = self.var["chr"]

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        # Retrieve sequence with a center length of seq_len from peak.
        chrom, start, end = self.var["chr"][index], self.var["start"][index], self.var["end"][index]
        mid = (int(start) + int(end)) // 2
        left, right = mid - self.seq_len//2, mid + self.seq_len//2
        left_pad, right_pad = 0, 0
        if left < 0:
            left_pad = -left_pad
            left = 0
        if right > self.genome[chrom].shape[0]:
            right_pad = right - self.genome[chrom].shape[0]
            right = self.genome[chrom].shape[0]
        seq = self.genome[chrom][left:right]
        # imputation
        if len(seq) < self.seq_len:
            seq = np.concatenate((
                np.full(left_pad, -1, dtype=seq.dtype),
                seq,
                np.full(right_pad, -1, dtype=seq.dtype),
            ))
        return seq, self.X[index].toarray().flatten()
