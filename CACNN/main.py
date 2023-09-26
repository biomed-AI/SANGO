import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import roc_auc_score
import dataset
import scanpy as sc
from model import CACNN
from utils import make_directory, make_logger, get_run_info
from utils import model_summary, set_seed
from sklearn.metrics import adjusted_rand_score

@torch.no_grad()
@autocast()
def test_model(model, loader, subset=None):
    model.eval()
    all_label = list()
    all_pred = list()

    for it, (seq, target) in enumerate(tqdm(loader)):
        seq = seq.to(device)
        output, _ = model(seq)
        output = output.detach()
        output = torch.sigmoid(output).cpu().numpy()
        target = target.numpy().astype(np.int8)
        all_pred.append(output)
        all_label.append(target)

    all_pred = np.concatenate(all_pred, axis=0) # (n_peaks, n_cells)
    all_label = np.concatenate(all_label, axis=0)
    val_auc = list()
    test_inds = range(all_pred.shape[0])
    for i in tqdm(test_inds, desc="Calculating AP"):
        val_auc.append(roc_auc_score(all_label[i], all_pred[i]))
    val_auc = np.array(val_auc)
    return val_auc

def get_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('-i', "--data", default="../preprocessed_data/BoneMarrowB_liver.h5ad", help="h5ad data")
    p.add_argument("-z", type=int, default=64)
    p.add_argument("-g", choices=("hg19", "hg38", "mm9", "mm10"), default="hg38")
    p.add_argument("-lr", type=float, default=1e-2)
    p.add_argument('-b', "--batch-size", help="batch size", type=int, default=128)
    p.add_argument("--num-workers", help="number of workers", type=int, default=32)
    p.add_argument("--seq-len", type=int, default=1344)
    p.add_argument("-o", "--outdir", help="output directory", default="output")
    p.add_argument("-w", help="load weight", action='store_true')
    p.add_argument('--seed', type=int, default=2020)
    p.add_argument('--max_epoch', type=int, default=300)
    p.add_argument('--device', type=int, default=0)
    p.add_argument("--use_reg_cell", action='store_true', help='use reg cell')
    p.add_argument("--alpha", type=float, default=0.0, help='weight of reg cell')
    return p



if __name__ == "__main__":
    args = get_args().parse_args()

    set_seed(args.seed)

    args.outdir = make_directory(args.outdir)
    logger = make_logger(title="", filename=os.path.join(args.outdir, "CACNN_train.log"))
    logger.info(get_run_info(sys.argv, args))

    if args.g == "hg38":
        genome = "../genome/GRCh38.primary_assembly.genome.fa.h5"
    elif args.g == "hg19":
        genome = "../genome/GRCh38.primary_assembly.genome.fa.h5"
    elif args.g == "mm9":
        genome = "../genome/mm9.fa.h5"
    elif args.g == "mm10":
        genome = "../genome/mm10.fa.h5"
    ds = dataset.SingleCellDataset(dataset.load_adata(args.data), seq_len=args.seq_len, genome=genome)

    train_loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4
    )
    
    sampled = np.random.permutation(np.arange(len(ds)))[:2000]
    valid_loader = DataLoader(
        Subset(ds, sampled),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    model = CACNN(n_cells=ds.X.shape[1], hidden_size=args.z, seq_len=args.seq_len, use_reg_cell=args.use_reg_cell).to(device)

    # train
    if not args.w:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.95, 0.9995))
        logger.info("{}\n{}\n{}\n".format(model, model_summary(model), optimizer))

        criterion = nn.BCEWithLogitsLoss()
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            mode="max",
            factor=0.9,
            patience=2,
            min_lr=1e-7
        )

        best_score = 0
        wait = 0
        patience = 30

        max_epoch = args.max_epoch
        for epoch in range(max_epoch):
            pool = [np.nan for _ in range(10)]
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epoch}")
            model.train()
            for it, (seq, target) in enumerate(pbar):
                seq = seq.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                with autocast():
                    output, lr_reg_cell = model(seq)
                    loss = criterion(output, target)
                    if(args.use_reg_cell):
                        loss += args.alpha * lr_reg_cell
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                pool[it % 10] = loss.item()

                lr = optimizer.param_groups[-1]["lr"]
                pbar.set_postfix_str(f"loss/lr={np.nanmean(pool):.4f}/{lr:.3e}")

            val_auc = test_model(model, valid_loader)
            logger.info("Validation{} AUC={:.4f}/{:.4f}".format((epoch + 1), val_auc.mean(), np.std(val_auc)))
            val_score = val_auc.mean()
            scheduler.step(val_score)
            if val_score > best_score:
                best_score = val_score
                wait = 0
                torch.save(model.state_dict(), "{}/CACNN_best_model.pt".format(args.outdir))
                logger.info(f"Epoch {epoch+1}: best model saved\n")
            else:
                wait += 1
                logger.info(f"Epoch {epoch+1}: early stopping patience {wait}/{patience}\n")
                if wait >= patience:
                    logger.info(f"Epoch {epoch+1}: early stopping")
                    break

    model.load_state_dict(torch.load("{}/CACNN_best_model.pt".format(args.outdir)))
    embedding = model.get_embedding().detach().cpu().numpy()
    logger.info("embedding: {}".format(embedding.shape))
    adata = sc.AnnData(
        embedding,
        obs=ds.obs,
    )
    sc.pp.neighbors(adata)
    sc.tl.umap(adata)
    sc.tl.leiden(adata)
    adata.write_h5ad("{}/CACNN_output.h5ad".format(args.outdir), compression="gzip")

    logger.info("ARI={}".format(adjusted_rand_score(adata.obs["CellType"], adata.obs["leiden"])))

